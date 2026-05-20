from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-06):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight

class SelectiveStateSpaceMixer(nn.Module):

    def __init__(self, d_model, d_state=64, conv_kernel=3, expand=2):
        super().__init__()
        inner_dim = d_model * expand
        self.in_proj = nn.Linear(d_model, inner_dim * 2)
        self.dw_conv = nn.Conv1d(inner_dim, inner_dim, kernel_size=conv_kernel, padding=conv_kernel - 1, groups=inner_dim)
        self.dt_proj = nn.Linear(inner_dim, d_state)
        self.B_proj = nn.Linear(inner_dim, d_state)
        self.C_proj = nn.Linear(inner_dim, d_state)
        self.state_to_inner = nn.Linear(d_state, inner_dim)
        self.D = nn.Parameter(torch.ones(inner_dim))
        self.out_proj = nn.Linear(inner_dim, d_model)
        self.act = nn.SiLU()

    def forward(self, x):
        (bsz, seq_len, _) = x.shape
        xz = self.in_proj(x)
        (x_main, gate) = xz.chunk(2, dim=-1)
        x_main = x_main.transpose(1, 2)
        x_main = self.dw_conv(x_main)[..., :seq_len]
        x_main = x_main.transpose(1, 2)
        x_main = self.act(x_main)
        dt = torch.sigmoid(self.dt_proj(x_main))
        B = self.B_proj(x_main)
        C = self.C_proj(x_main)
        state = x.new_zeros(bsz, B.size(-1))
        outputs = []
        for t in range(seq_len):
            state = (1.0 - dt[:, t]) * state + dt[:, t] * B[:, t]
            yt = C[:, t] * state
            outputs.append(yt)
        y = torch.stack(outputs, dim=1)
        y = F.layer_norm(y, (y.size(-1),))
        y = self.state_to_inner(y)
        y = y + self.D * x_main
        y = y * torch.sigmoid(gate)
        return self.out_proj(y)

class OfficialMambaMixer(nn.Module):

    def __init__(self, d_model, d_state=64, conv_kernel=3, expand=2):
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except ImportError as e:
            raise ImportError('mamba-ssm is not installed. Install with: `pip install mamba-ssm --no-build-isolation`') from e
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=conv_kernel, expand=expand)

    def forward(self, x):
        return self.mamba(x)

class MoEFeedForward(nn.Module):

    def __init__(self, d_model, ff_mult=4, num_experts=4, top_k=2, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        hidden = d_model * ff_mult
        self.gate = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, d_model)) for _ in range(num_experts)])

    def forward(self, x):
        logits = self.gate(x)
        (top_vals, top_idx) = torch.topk(logits, k=self.top_k, dim=-1)
        top_probs = torch.softmax(top_vals, dim=-1)
        out = torch.zeros_like(x)
        for (expert_id, expert) in enumerate(self.experts):
            expert_out = expert(x)
            match = (top_idx == expert_id).float()
            weight = (top_probs * match).sum(dim=-1, keepdim=True)
            out = out + expert_out * weight
        return out

class MixtureOfMambaBlock(nn.Module):

    def __init__(self, d_model, d_state=64, conv_kernel=3, expand=2, ff_mult=4, num_experts=4, top_k=2, dropout=0.1, mixer_backend='custom'):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        if mixer_backend == 'official':
            self.mixer = OfficialMambaMixer(d_model=d_model, d_state=d_state, conv_kernel=conv_kernel, expand=expand)
        else:
            self.mixer = SelectiveStateSpaceMixer(d_model=d_model, d_state=d_state, conv_kernel=conv_kernel, expand=expand)
        self.norm2 = RMSNorm(d_model)
        self.moe = MoEFeedForward(d_model=d_model, ff_mult=ff_mult, num_experts=num_experts, top_k=top_k, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.mixer(self.norm1(x)))
        x = x + self.dropout(self.moe(self.norm2(x)))
        return x

class MixtureOfMambaModel(nn.Module):

    def __init__(self, num_classes=13, d_model=768, hidden_dim=None, n_layers=4, d_state=64, conv_kernel=3, expand=2, ff_mult=4, num_experts=4, top_k=2, dropout=0.1, vocab_size=5000, max_video_tokens=32, max_audio_tokens=64, use_official_mamba_ssm=True, hf_pretrained_mamba_model='state-spaces/mamba-130m-hf', auto_load_pretrained_mamba=False, pretrained_mamba_path=None, pretrained_mamba_prefix='', freeze_mamba=False, **_):
        super().__init__()
        if hidden_dim is not None:
            d_model = int(hidden_dim)
        self.max_video_tokens = max_video_tokens
        self.max_audio_tokens = max_audio_tokens
        self.image_stem = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(64), nn.GELU(), nn.AdaptiveAvgPool2d((1, 1)))
        self.video_proj = nn.LazyLinear(d_model)
        self.audio_proj = nn.LazyLinear(d_model)
        self.question_embedding = nn.Embedding(vocab_size, d_model)
        self.question_proj = nn.Linear(d_model, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.modality_embed = nn.Parameter(torch.zeros(1, 4, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + max_video_tokens + max_audio_tokens + 1, d_model))
        if use_official_mamba_ssm:
            try:
                import mamba_ssm
            except ImportError as e:
                raise ImportError('Official Mamba backend is enabled by default but `mamba-ssm` is missing. Install with: `pip install mamba-ssm --no-build-isolation`') from e
        self.blocks = nn.ModuleList([MixtureOfMambaBlock(d_model=d_model, d_state=d_state, conv_kernel=conv_kernel, expand=expand, ff_mult=ff_mult, num_experts=num_experts, top_k=top_k, dropout=dropout, mixer_backend='official' if use_official_mamba_ssm else 'custom') for _ in range(n_layers)])
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.modality_embed, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if hf_pretrained_mamba_model:
            self._load_pretrained_mamba_from_hf(hf_pretrained_mamba_model)
        auto_path = None
        if auto_load_pretrained_mamba and (not pretrained_mamba_path):
            auto_path = self._find_default_mamba_checkpoint()
        effective_pretrained_path = pretrained_mamba_path or auto_path
        if effective_pretrained_path:
            self._load_pretrained_mamba(pretrained_path=effective_pretrained_path, prefix=pretrained_mamba_prefix)
        elif auto_load_pretrained_mamba:
            print('[MixtureOfMambaModel] Auto pretrained Mamba checkpoint was not found. Training starts from random initialization for Mamba blocks.')
        if freeze_mamba:
            self._freeze_mamba_blocks()

    def _find_default_mamba_checkpoint(self):
        project_root = Path(__file__).resolve().parents[1]
        candidates = [project_root / 'checkpoints' / 'mixture_of_mamba' / 'model_best.pth', project_root / 'checkpoints' / 'mixture_of_mamba' / 'model_last.pth', project_root / 'checkpoints' / 'mixture_of_mamba_advance' / 'model_best.pth', project_root / 'checkpoints' / 'mixture_of_mamba_advance' / 'model_last.pth', project_root / 'checkpoints' / 'mamba' / 'model_best.pth', project_root / 'checkpoints' / 'mamba' / 'model_last.pth']
        for path in candidates:
            if path.exists():
                return str(path)
        return None

    def _freeze_mamba_blocks(self):
        for p in self.blocks.parameters():
            p.requires_grad = False
        if hasattr(self, 'norm'):
            for p in self.norm.parameters():
                p.requires_grad = False

    def _load_pretrained_mamba(self, pretrained_path, prefix=''):
        ckpt = torch.load(pretrained_path, map_location='cpu')
        src_state = ckpt.get('state_dict', ckpt)
        if not isinstance(src_state, dict):
            print(f"[MixtureOfMambaModel] Unexpected checkpoint format at {pretrained_path}. Expected dict or dict with 'state_dict'. Skipping pretrained load.")
            return
        if prefix:
            normalized = {}
            pref = prefix if prefix.endswith('.') else f'{prefix}.'
            for (k, v) in src_state.items():
                if k.startswith(pref):
                    normalized[k[len(pref):]] = v
            src_state = normalized
        dst_state = self.state_dict()
        allowed_prefixes = ('blocks.', 'norm.')
        to_load = {}
        for (k, v) in src_state.items():
            if not k.startswith(allowed_prefixes):
                continue
            if k in dst_state and dst_state[k].shape == v.shape:
                to_load[k] = v
        if not to_load:
            print('[MixtureOfMambaModel] No matching Mamba block weights found. Check checkpoint key names/shapes and pretrained_mamba_prefix.')
            return
        self.load_state_dict(to_load, strict=False)
        print(f'[MixtureOfMambaModel] Loaded {len(to_load)} tensors from pretrained Mamba checkpoint: {pretrained_path}')

    def _load_pretrained_mamba_from_hf(self, hf_model_name):
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as e:
            raise ImportError('transformers is required for HF Mamba loading. Install with `pip install transformers`.') from e
        if not any((isinstance(block.mixer, OfficialMambaMixer) for block in self.blocks)):
            print('[MixtureOfMambaModel] HF Mamba weight loading expects official mamba-ssm backend. Skipping because current model uses custom mixer.')
            return
        src_state = None
        try:
            hf_model = AutoModelForCausalLM.from_pretrained(hf_model_name, trust_remote_code=True)
            src_state = hf_model.state_dict()
        except Exception as e:
            print(f'[MixtureOfMambaModel] AutoModelForCausalLM loading failed, trying direct safetensors download from Hugging Face. Original error: {e}')
            src_state = self._load_hf_state_dict_without_model_init(hf_model_name)
            if src_state is None:
                print('[MixtureOfMambaModel] Failed to load HF pretrained weights; continuing with random initialization.')
                return
        dst_state = self.state_dict()
        mapped = {}
        layer_idx = 0
        while layer_idx < len(self.blocks):
            src_prefix = f'backbone.layers.{layer_idx}.mixer.'
            dst_prefix = f'blocks.{layer_idx}.mixer.mamba.'
            layer_has_any = False
            for (src_key, value) in src_state.items():
                if not src_key.startswith(src_prefix):
                    continue
                layer_has_any = True
                dst_key = dst_prefix + src_key[len(src_prefix):]
                if dst_key in dst_state and dst_state[dst_key].shape == value.shape:
                    mapped[dst_key] = value
            if not layer_has_any:
                break
            layer_idx += 1
        if 'backbone.norm_f.weight' in src_state and 'norm.weight' in dst_state:
            if src_state['backbone.norm_f.weight'].shape == dst_state['norm.weight'].shape:
                mapped['norm.weight'] = src_state['backbone.norm_f.weight']
        if not mapped:
            print('[MixtureOfMambaModel] Could not map any weights from HF Mamba model. Likely hidden size mismatch (e.g., HF 768/1024/...) vs model d_model.')
            return
        self.load_state_dict(mapped, strict=False)
        print(f'[MixtureOfMambaModel] Loaded {len(mapped)} tensors from HF model: {hf_model_name}')

    def _load_hf_state_dict_without_model_init(self, hf_model_name):
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            print('[MixtureOfMambaModel] `huggingface_hub` is missing; cannot download HF checkpoint tensors directly.')
            return None
        try:
            safetensors_path = hf_hub_download(repo_id=hf_model_name, filename='model.safetensors')
            try:
                from safetensors.torch import load_file as safe_load_file
            except ImportError:
                print('[MixtureOfMambaModel] `safetensors` is missing; cannot read model.safetensors.')
                return None
            return safe_load_file(safetensors_path)
        except Exception:
            pass
        try:
            bin_path = hf_hub_download(repo_id=hf_model_name, filename='pytorch_model.bin')
            return torch.load(bin_path, map_location='cpu')
        except Exception as e:
            print(f'[MixtureOfMambaModel] Could not download HF checkpoint tensors from {hf_model_name}. Error: {e}')
            return None

    def _encode_video(self, video):
        (bsz, timesteps) = video.shape[:2]
        if video.dim() != 5:
            raise ValueError(f'Expected video shape [B, T, C, H, W], got {tuple(video.shape)}')
        c = video.size(2)
        if c == 3:
            frames = video.view(bsz * timesteps, c, video.size(3), video.size(4))
            feat = self.image_stem(frames).flatten(1)
            feat = feat.view(bsz, timesteps, -1)
        else:
            feat = video.mean(dim=(-1, -2))
        if feat.size(1) > self.max_video_tokens:
            feat = feat[:, :self.max_video_tokens]
        return self.video_proj(feat)

    def _encode_audio(self, audio):
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        if audio.dim() != 3:
            raise ValueError(f'Expected audio shape [B, T, F], got {tuple(audio.shape)}')
        if audio.size(1) > self.max_audio_tokens:
            audio = audio[:, :self.max_audio_tokens]
        return self.audio_proj(audio)

    def _encode_question(self, question, batch_size, device):
        if question is None:
            return torch.zeros(batch_size, 1, self.question_proj.out_features, device=device)
        if question.dim() == 1:
            question = question.unsqueeze(1)
        if question.dim() != 2:
            question = question.view(batch_size, -1)
        q = self.question_embedding(question.long().clamp(min=0))
        q = self.question_proj(q.mean(dim=1, keepdim=True))
        return q

    def forward(self, video, audio, question=None):
        bsz = video.size(0)
        device = video.device
        v_tokens = self._encode_video(video)
        a_tokens = self._encode_audio(audio)
        q_token = self._encode_question(question, bsz, device)
        cls = self.cls_token.expand(bsz, -1, -1)
        v_tokens = v_tokens + self.modality_embed[:, 0:1]
        a_tokens = a_tokens + self.modality_embed[:, 1:2]
        q_token = q_token + self.modality_embed[:, 2:3]
        cls = cls + self.modality_embed[:, 3:4]
        x = torch.cat([v_tokens, a_tokens, q_token, cls], dim=1)
        x = x + self.pos_embed[:, :x.size(1)]
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x[:, -1])
        return self.head(x)

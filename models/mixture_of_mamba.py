from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from transformers import Mask2FormerConfig, Mask2FormerModel
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

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
        bsz, seq_len, _ = x.shape
        xz = self.in_proj(x)
        x_main, gate = xz.chunk(2, dim=-1)
        x_main = self.dw_conv(x_main.transpose(1, 2))[..., :seq_len].transpose(1, 2)
        x_main = self.act(x_main)
        dt = torch.sigmoid(self.dt_proj(x_main))
        B = self.B_proj(x_main)
        C = self.C_proj(x_main)
        state = x.new_zeros(bsz, B.size(-1))
        outputs = []
        for t in range(seq_len):
            state = (1.0 - dt[:, t]) * state + dt[:, t] * B[:, t]
            outputs.append(C[:, t] * state)
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

class MambaSequenceBlock(nn.Module):

    def __init__(self, d_model, d_state=16, conv_kernel=4, expand=2, ff_mult=4, dropout=0.1, use_official_mamba_ssm=True):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.mixer = OfficialMambaMixer(d_model=d_model, d_state=d_state, conv_kernel=conv_kernel, expand=expand) if use_official_mamba_ssm else SelectiveStateSpaceMixer(d_model=d_model, d_state=d_state, conv_kernel=conv_kernel, expand=expand)
        self.norm2 = RMSNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * ff_mult), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * ff_mult, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.mixer(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

class MambaPretrainedMixin:

    def _load_pretrained_mamba_from_hf(self, hf_model_name, containers):
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as e:
            raise ImportError('transformers is required for HF Mamba loading.') from e
        try:
            src_state = AutoModelForCausalLM.from_pretrained(hf_model_name, trust_remote_code=True).state_dict()
        except Exception as e:
            print(f'[{type(self).__name__}] HF Mamba load failed ({e}); continuing with random Mamba init.')
            return
        dst_state = self.state_dict()
        mapped = {}
        src_layer_idx = 0
        for container_name in containers:
            container = getattr(self, container_name, None)
            if container is None:
                continue
            for dst_layer_idx in range(len(container)):
                src_prefix = f'backbone.layers.{src_layer_idx}.mixer.'
                dst_prefix = f'{container_name}.{dst_layer_idx}.mixer.mamba.'
                layer_has_any = False
                for src_key, value in src_state.items():
                    if not src_key.startswith(src_prefix):
                        continue
                    layer_has_any = True
                    dst_key = dst_prefix + src_key[len(src_prefix):]
                    if dst_key in dst_state and dst_state[dst_key].shape == value.shape:
                        mapped[dst_key] = value
                if not layer_has_any:
                    break
                src_layer_idx += 1
        if mapped:
            self.load_state_dict(mapped, strict=False)
            print(f'[{type(self).__name__}] Loaded {len(mapped)} tensors from HF Mamba: {hf_model_name}')

    def _load_pretrained_mamba_checkpoint(self, pretrained_path, prefix='', allowed_prefixes=()):
        ckpt = torch.load(pretrained_path, map_location='cpu')
        src_state = ckpt.get('state_dict', ckpt)
        if not isinstance(src_state, dict):
            return
        if prefix:
            pref = prefix if prefix.endswith('.') else f'{prefix}.'
            src_state = {k[len(pref):]: v for k, v in src_state.items() if k.startswith(pref)}
        dst_state = self.state_dict()
        to_load = {k: v for k, v in src_state.items() if (not allowed_prefixes or k.startswith(allowed_prefixes)) and k in dst_state and dst_state[k].shape == v.shape}
        if to_load:
            self.load_state_dict(to_load, strict=False)
            print(f'[{type(self).__name__}] Loaded {len(to_load)} tensors from {pretrained_path}')

class Mask2FormerFrameEncoder(nn.Module):

    def __init__(self, d_model, pretrained_visual_model='facebook/mask2former-swin-base-ade-semantic', freeze_visual_backbone=True):
        super().__init__()
        if not _HF_AVAILABLE:
            raise ImportError('transformers is required for Mask2FormerFrameEncoder.')
        loaded_pretrained = False
        if pretrained_visual_model is not None:
            try:
                model = Mask2FormerModel.from_pretrained(pretrained_visual_model)
                loaded_pretrained = True
            except Exception as e:
                print(f"[Mask2FormerFrameEncoder] Failed to load pretrained Mask2Former ('{pretrained_visual_model}'): {e}. Falling back to random init.")
                model = Mask2FormerModel(Mask2FormerConfig())
        else:
            model = Mask2FormerModel(Mask2FormerConfig())
        self.encoder = model.pixel_level_module.encoder
        del model
        if freeze_visual_backbone and loaded_pretrained:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self.proj = nn.LazyConv2d(d_model, kernel_size=1)

    def forward(self, frames):
        b, t, c, h, w = frames.shape
        x = frames.view(b * t, c, h, w)
        frozen = not next(self.encoder.parameters()).requires_grad
        if frozen:
            with torch.no_grad():
                out = self.encoder(pixel_values=x)
        else:
            out = self.encoder(pixel_values=x)
        fmap = out.feature_maps[-1]
        if fmap.dim() == 3:
            s = int(fmap.size(1) ** 0.5)
            fmap = fmap.transpose(1, 2).contiguous().view(fmap.size(0), fmap.size(2), s, s)
        elif fmap.dim() == 4 and fmap.size(1) < fmap.size(-1):
            fmap = fmap.permute(0, 3, 1, 2).contiguous()
        spatial = self.proj(fmap)
        tokens = spatial.mean(dim=(-1, -2)).view(b, t, -1)
        return spatial.view(b, t, *spatial.shape[1:]), tokens

class MaskDecoder(nn.Module):

    def __init__(self, d_model, mask_size):
        super().__init__()
        self.mask_size = int(mask_size)
        self.fuse = nn.Sequential(nn.Conv2d(d_model * 2, d_model, kernel_size=3, padding=1), nn.GELU(), nn.Conv2d(d_model, d_model // 2, kernel_size=3, padding=1), nn.GELU(), nn.Conv2d(d_model // 2, 1, kernel_size=1))

    def forward(self, query, spatial):
        query_map = query.unsqueeze(-1).unsqueeze(-1).expand_as(spatial)
        logits = self.fuse(torch.cat([spatial, query_map], dim=1))
        logits = F.interpolate(logits, size=(self.mask_size, self.mask_size), mode='bilinear', align_corners=False)
        return logits.squeeze(1)

class BaseAVSceneModel(nn.Module, MambaPretrainedMixin):

    def __init__(self, d_model=768, audio_dim=128, text_dim=768, max_audio_tokens=10, max_text_tokens=25, mask_size=256, pretrained_visual_model='facebook/mask2former-swin-base-ade-semantic', freeze_visual_backbone=True, dropout=0.1):
        super().__init__()
        self.max_audio_tokens = int(max_audio_tokens)
        self.max_text_tokens = int(max_text_tokens)
        self.text_dim = int(text_dim)
        self.visual_encoder = Mask2FormerFrameEncoder(d_model=d_model, pretrained_visual_model=pretrained_visual_model, freeze_visual_backbone=freeze_visual_backbone)
        self.audio_proj = nn.Linear(audio_dim, d_model)
        self.text_proj = nn.Linear(text_dim, d_model)
        self.decoder = MaskDecoder(d_model=d_model, mask_size=mask_size)
        self.dropout = nn.Dropout(dropout)

    def _project_inputs(self, frames, audio_feat, text_feat=None):
        if text_feat is None:
            text_feat = frames.new_zeros(frames.size(0), 1, self.text_dim)
        spatial, video_tokens = self.visual_encoder(frames)
        audio_tokens = self.audio_proj(audio_feat[:, :self.max_audio_tokens])
        text_tokens = self.text_proj(text_feat[:, :self.max_text_tokens])
        return spatial, video_tokens, audio_tokens, text_tokens

    def _decode(self, queries, spatial):
        logits = []
        for idx in range(queries.size(1)):
            logits.append(self.decoder(queries[:, idx], spatial[:, idx]))
        return torch.cat(logits, dim=0)

class MoEFeedForward(nn.Module):

    def __init__(self, d_model, ff_mult=4, num_experts=4, top_k=2, dropout=0.1):
        super().__init__()
        self.top_k = min(top_k, num_experts)
        hidden = d_model * ff_mult
        self.gate = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, d_model)) for _ in range(num_experts)])

    def forward(self, x):
        logits = self.gate(x)
        top_vals, top_idx = torch.topk(logits, k=self.top_k, dim=-1)
        top_probs = torch.softmax(top_vals, dim=-1)
        out = torch.zeros_like(x)
        for expert_id, expert in enumerate(self.experts):
            expert_out = expert(x)
            match = (top_idx == expert_id).float()
            weight = (top_probs * match).sum(dim=-1, keepdim=True)
            out = out + expert_out * weight
        return out

class MixtureOfMambaBlock(nn.Module):

    def __init__(self, d_model, d_state=64, conv_kernel=3, expand=2, ff_mult=4, num_experts=4, top_k=2, dropout=0.1, use_official_mamba_ssm=True):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.mixer = OfficialMambaMixer(d_model=d_model, d_state=d_state, conv_kernel=conv_kernel, expand=expand) if use_official_mamba_ssm else SelectiveStateSpaceMixer(d_model=d_model, d_state=d_state, conv_kernel=conv_kernel, expand=expand)
        self.norm2 = RMSNorm(d_model)
        self.moe = MoEFeedForward(d_model=d_model, ff_mult=ff_mult, num_experts=num_experts, top_k=top_k, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.mixer(self.norm1(x)))
        x = x + self.dropout(self.moe(self.norm2(x)))
        return x

class MixtureOfMambaModel(BaseAVSceneModel):

    def __init__(self, d_model=768, hidden_dim=None, n_layers=4, d_state=16, conv_kernel=4, expand=2, ff_mult=4, num_experts=4, top_k=2, dropout=0.1, audio_dim=128, text_dim=768, max_audio_tokens=10, max_text_tokens=25, mask_size=256, image_size=256, pretrained_visual_model='facebook/mask2former-swin-base-ade-semantic', freeze_visual_backbone=True, use_official_mamba_ssm=True, hf_pretrained_mamba_model='state-spaces/mamba-130m-hf', auto_load_pretrained_mamba=False, pretrained_mamba_path=None, pretrained_mamba_prefix='', freeze_mamba=False, **_):
        if hidden_dim is not None:
            d_model = int(hidden_dim)
        super().__init__(d_model=d_model, audio_dim=audio_dim, text_dim=text_dim, max_audio_tokens=max_audio_tokens, max_text_tokens=max_text_tokens, mask_size=mask_size, pretrained_visual_model=pretrained_visual_model, freeze_visual_backbone=freeze_visual_backbone, dropout=dropout)
        self.blocks = nn.ModuleList([MixtureOfMambaBlock(d_model=d_model, d_state=d_state, conv_kernel=conv_kernel, expand=expand, ff_mult=ff_mult, num_experts=num_experts, top_k=top_k, dropout=dropout, use_official_mamba_ssm=use_official_mamba_ssm) for _ in range(n_layers)])
        self.blocks_norm = RMSNorm(d_model)
        self.modality_embed = nn.Parameter(torch.zeros(1, 4, d_model))
        self.mask_query = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 10 + max_audio_tokens + max_text_tokens + 10, d_model))
        nn.init.trunc_normal_(self.modality_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_query, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if hf_pretrained_mamba_model:
            self._load_pretrained_mamba_from_hf(hf_pretrained_mamba_model, ('blocks',))
        if pretrained_mamba_path:
            self._load_pretrained_mamba_checkpoint(pretrained_mamba_path, pretrained_mamba_prefix, ('blocks.', 'blocks_norm.'))
        if freeze_mamba:
            for p in self.blocks.parameters():
                p.requires_grad = False
            for p in self.blocks_norm.parameters():
                p.requires_grad = False

    def forward(self, frames, audio_feat, text_feat=None, masks=None, **_):
        del masks
        spatial, video_tokens, audio_tokens, text_tokens = self._project_inputs(frames, audio_feat, text_feat)
        b, t = video_tokens.shape[:2]
        queries = self.mask_query.expand(b, t, -1)
        x = torch.cat([video_tokens + self.modality_embed[:, 0:1], audio_tokens + self.modality_embed[:, 1:2], text_tokens + self.modality_embed[:, 2:3], queries + self.modality_embed[:, 3:4]], dim=1)
        x = self.dropout(x + self.pos_embed[:, :x.size(1)])
        for block in self.blocks:
            x = block(x)
        x = self.blocks_norm(x)
        q_start = t + audio_tokens.size(1) + text_tokens.size(1)
        queries = x[:, q_start:q_start + t]
        return {'logits': self._decode(queries, spatial)}

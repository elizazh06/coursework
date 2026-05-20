from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mixture_of_mamba import MixtureOfMambaBlock, RMSNorm

class _MaskDecoder(nn.Module):

    def __init__(self, d_model: int, mask_size: int):
        super().__init__()
        self.mask_size = mask_size
        self.fuse = nn.Sequential(nn.Conv2d(d_model * 2, d_model, kernel_size=3, padding=1), nn.GELU(), nn.Conv2d(d_model, d_model // 2, kernel_size=3, padding=1), nn.GELU(), nn.Conv2d(d_model // 2, 1, kernel_size=1))

    def forward(self, token: torch.Tensor, spatial: torch.Tensor) -> torch.Tensor:
        token_map = token.unsqueeze(-1).unsqueeze(-1).expand_as(spatial)
        x = torch.cat([spatial, token_map], dim=1)
        x = self.fuse(x)
        x = F.interpolate(x, size=(self.mask_size, self.mask_size), mode='bilinear', align_corners=False)
        return x.squeeze(1)

class MixtureOfMambaRefAVSModel(nn.Module):

    def __init__(self, d_model: int=768, n_layers: int=4, d_state: int=16, conv_kernel: int=4, expand: int=2, ff_mult: int=4, num_experts: int=4, top_k: int=2, dropout: float=0.1, audio_dim: int=128, text_dim: int=768, max_text_tokens: int=25, mask_size: int=256, image_size: int=256, use_official_mamba_ssm: bool=True, hf_pretrained_mamba_model: str | None='state-spaces/mamba-130m-hf', pretrained_mamba_path: str | None=None, pretrained_mamba_prefix: str='', freeze_mamba: bool=False, **_):
        super().__init__()
        self.mask_size = int(mask_size)
        self.max_text_tokens = int(max_text_tokens)
        self.frame_encoder = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(64), nn.GELU(), nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU(), nn.Conv2d(128, d_model, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(d_model), nn.GELU())
        self.audio_proj = nn.Linear(audio_dim, d_model)
        self.text_proj = nn.Linear(text_dim, d_model)
        self.modality_embed = nn.Parameter(torch.zeros(1, 4, d_model))
        self.mask_query = nn.Parameter(torch.zeros(1, 1, d_model))
        max_seq = 10 + 10 + max_text_tokens + 10
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq, d_model))
        if use_official_mamba_ssm:
            try:
                import mamba_ssm
            except ImportError as e:
                raise ImportError('Official Mamba backend is enabled but `mamba-ssm` is missing. Install with: `pip install mamba-ssm --no-build-isolation`') from e
        mixer_backend = 'official' if use_official_mamba_ssm else 'custom'
        self.blocks = nn.ModuleList([MixtureOfMambaBlock(d_model=d_model, d_state=d_state, conv_kernel=conv_kernel, expand=expand, ff_mult=ff_mult, num_experts=num_experts, top_k=top_k, dropout=dropout, mixer_backend=mixer_backend) for _ in range(n_layers)])
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.mask_decoder = _MaskDecoder(d_model, mask_size)
        nn.init.trunc_normal_(self.modality_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_query, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if hf_pretrained_mamba_model:
            self._load_pretrained_mamba_from_hf(hf_pretrained_mamba_model)
        if pretrained_mamba_path:
            self._load_pretrained_mamba(pretrained_mamba_path, pretrained_mamba_prefix)
        if freeze_mamba:
            for p in self.blocks.parameters():
                p.requires_grad = False
            for p in self.norm.parameters():
                p.requires_grad = False

    def _encode_frames(self, frames: torch.Tensor):
        (b, t, c, h, w) = frames.shape
        x = frames.view(b * t, c, h, w)
        spatial = self.frame_encoder(x)
        tokens = spatial.mean(dim=(-1, -2)).view(b, t, -1)
        return (spatial, tokens)

    def _build_sequence(self, video_tokens: torch.Tensor, audio_tokens: torch.Tensor, text_tokens: torch.Tensor, n_frames: int) -> torch.Tensor:
        b = video_tokens.size(0)
        t_a = audio_tokens.size(1)
        t_t = min(text_tokens.size(1), self.max_text_tokens)
        text_tokens = text_tokens[:, :t_t]
        mask_q = self.mask_query.expand(b, n_frames, -1)
        v = video_tokens + self.modality_embed[:, 0:1]
        a = audio_tokens + self.modality_embed[:, 1:2]
        txt = text_tokens + self.modality_embed[:, 2:3]
        mq = mask_q + self.modality_embed[:, 3:4]
        x = torch.cat([v, a, txt, mq], dim=1)
        x = x + self.pos_embed[:, :x.size(1)]
        return x

    def forward(self, frames: torch.Tensor, audio_feat: torch.Tensor, text_feat: torch.Tensor, masks: torch.Tensor | None=None, **_) -> dict:
        del masks
        (b, t, _, _, _) = frames.shape
        (spatial, video_tokens) = self._encode_frames(frames)
        audio_tokens = self.audio_proj(audio_feat)
        text_tokens = self.text_proj(text_feat)
        x = self._build_sequence(video_tokens, audio_tokens, text_tokens, t)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        t_a = audio_tokens.size(1)
        t_t = min(text_tokens.size(1), self.max_text_tokens)
        mq_start = t + t_a + t_t
        mq_states = x[:, mq_start:mq_start + t]
        spatial_maps = spatial.view(b, t, *spatial.shape[1:])
        logits_list = []
        for f in range(t):
            logits_list.append(self.mask_decoder(mq_states[:, f], spatial_maps[:, f]))
        logits = torch.cat(logits_list, dim=0)
        return {'logits': logits}

    def _load_pretrained_mamba(self, pretrained_path: str, prefix: str=''):
        ckpt = torch.load(pretrained_path, map_location='cpu')
        src_state = ckpt.get('state_dict', ckpt)
        if not isinstance(src_state, dict):
            return
        if prefix:
            pref = prefix if prefix.endswith('.') else f'{prefix}.'
            src_state = {k[len(pref):]: v for (k, v) in src_state.items() if k.startswith(pref)}
        dst_state = self.state_dict()
        to_load = {k: v for (k, v) in src_state.items() if k.startswith(('blocks.', 'norm.')) and k in dst_state and (dst_state[k].shape == v.shape)}
        if to_load:
            self.load_state_dict(to_load, strict=False)
            print(f'[MixtureOfMambaRefAVSModel] Loaded {len(to_load)} tensors from {pretrained_path}')

    def _load_pretrained_mamba_from_hf(self, hf_model_name: str):
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as e:
            raise ImportError('transformers is required for HF Mamba loading.') from e
        try:
            hf_model = AutoModelForCausalLM.from_pretrained(hf_model_name, trust_remote_code=True)
            src_state = hf_model.state_dict()
        except Exception as e:
            print(f'[MixtureOfMambaRefAVSModel] HF load failed ({e}); continuing with random Mamba init.')
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
        if mapped:
            self.load_state_dict(mapped, strict=False)
            print(f'[MixtureOfMambaRefAVSModel] Loaded {len(mapped)} tensors from HF: {hf_model_name}')

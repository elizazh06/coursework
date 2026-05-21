from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mixture_of_mamba import OfficialMambaMixer, RMSNorm, SelectiveStateSpaceMixer

try:
    from transformers import Mask2FormerConfig, Mask2FormerModel
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False


class MambaSequenceBlock(nn.Module):

    def __init__(self, d_model: int, d_state: int=16, conv_kernel: int=4, expand: int=2, ff_mult: int=4, dropout: float=0.1, use_official_mamba_ssm: bool=True):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        if use_official_mamba_ssm:
            self.mixer = OfficialMambaMixer(d_model=d_model, d_state=d_state, conv_kernel=conv_kernel, expand=expand)
        else:
            self.mixer = SelectiveStateSpaceMixer(d_model=d_model, d_state=d_state, conv_kernel=conv_kernel, expand=expand)
        self.norm2 = RMSNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * ff_mult), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * ff_mult, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.mixer(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class MambaPretrainedMixin:

    def _load_pretrained_mamba_from_hf(self, hf_model_name: str, containers: tuple[str, ...]):
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as e:
            raise ImportError('transformers is required for HF Mamba loading.') from e
        try:
            hf_model = AutoModelForCausalLM.from_pretrained(hf_model_name, trust_remote_code=True)
            src_state = hf_model.state_dict()
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
        for container_name in containers:
            norm_name = f'{container_name}_norm'
            if hasattr(self, norm_name) and 'backbone.norm_f.weight' in src_state:
                dst_key = f'{norm_name}.weight'
                if dst_key in dst_state and dst_state[dst_key].shape == src_state['backbone.norm_f.weight'].shape:
                    mapped[dst_key] = src_state['backbone.norm_f.weight']
                    break
        if mapped:
            self.load_state_dict(mapped, strict=False)
            print(f'[{type(self).__name__}] Loaded {len(mapped)} tensors from HF Mamba: {hf_model_name}')
        else:
            print(f'[{type(self).__name__}] Could not map HF Mamba weights from {hf_model_name}.')

    def _load_pretrained_mamba_checkpoint(self, pretrained_path: str, prefix: str='', allowed_prefixes: tuple[str, ...]=()):
        ckpt = torch.load(pretrained_path, map_location='cpu')
        src_state = ckpt.get('state_dict', ckpt)
        if not isinstance(src_state, dict):
            return
        if prefix:
            pref = prefix if prefix.endswith('.') else f'{prefix}.'
            src_state = {k[len(pref):]: v for k, v in src_state.items() if k.startswith(pref)}
        dst_state = self.state_dict()
        to_load = {}
        for k, v in src_state.items():
            if allowed_prefixes and not k.startswith(allowed_prefixes):
                continue
            if k in dst_state and dst_state[k].shape == v.shape:
                to_load[k] = v
        if to_load:
            self.load_state_dict(to_load, strict=False)
            print(f'[{type(self).__name__}] Loaded {len(to_load)} tensors from {pretrained_path}')


class Mask2FormerFrameEncoder(nn.Module):

    def __init__(self, d_model: int, pretrained_visual_model: str | None='facebook/mask2former-swin-base-ade-semantic', freeze_visual_backbone: bool=True):
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

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

    def __init__(self, d_model: int, mask_size: int):
        super().__init__()
        self.mask_size = int(mask_size)
        self.fuse = nn.Sequential(nn.Conv2d(d_model * 2, d_model, kernel_size=3, padding=1), nn.GELU(), nn.Conv2d(d_model, d_model // 2, kernel_size=3, padding=1), nn.GELU(), nn.Conv2d(d_model // 2, 1, kernel_size=1))

    def forward(self, query: torch.Tensor, spatial: torch.Tensor) -> torch.Tensor:
        query_map = query.unsqueeze(-1).unsqueeze(-1).expand_as(spatial)
        logits = self.fuse(torch.cat([spatial, query_map], dim=1))
        logits = F.interpolate(logits, size=(self.mask_size, self.mask_size), mode='bilinear', align_corners=False)
        return logits.squeeze(1)


class BaseSSMAVSModel(nn.Module, MambaPretrainedMixin):

    def __init__(self, d_model: int=768, audio_dim: int=128, text_dim: int=768, max_audio_tokens: int=10, max_text_tokens: int=25, mask_size: int=256, pretrained_visual_model: str | None='facebook/mask2former-swin-base-ade-semantic', freeze_visual_backbone: bool=True, dropout: float=0.1):
        super().__init__()
        self.d_model = int(d_model)
        self.max_audio_tokens = int(max_audio_tokens)
        self.max_text_tokens = int(max_text_tokens)
        self.visual_encoder = Mask2FormerFrameEncoder(d_model=d_model, pretrained_visual_model=pretrained_visual_model, freeze_visual_backbone=freeze_visual_backbone)
        self.audio_proj = nn.Linear(audio_dim, d_model)
        self.text_proj = nn.Linear(text_dim, d_model)
        self.decoder = MaskDecoder(d_model=d_model, mask_size=mask_size)
        self.dropout = nn.Dropout(dropout)

    def _project_inputs(self, frames: torch.Tensor, audio_feat: torch.Tensor, text_feat: torch.Tensor):
        spatial, video_tokens = self.visual_encoder(frames)
        audio_tokens = self.audio_proj(audio_feat[:, :self.max_audio_tokens])
        text_tokens = self.text_proj(text_feat[:, :self.max_text_tokens])
        return spatial, video_tokens, audio_tokens, text_tokens

    def _decode(self, queries: torch.Tensor, spatial: torch.Tensor) -> torch.Tensor:
        b, t = queries.shape[:2]
        logits = []
        for idx in range(t):
            logits.append(self.decoder(queries[:, idx], spatial[:, idx]))
        return torch.cat(logits, dim=0)

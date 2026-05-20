from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from transformers import Mask2FormerModel, Mask2FormerConfig
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

class _CrossAttnDecoder(nn.Module):

    def __init__(self, dim_v: int, num_heads: int, mask_size: int, spatial_size: int, prompt_dim: int):
        super().__init__()
        self.spatial_size = spatial_size
        self.mask_size = mask_size
        self.prompt_proj = nn.Linear(prompt_dim, dim_v) if prompt_dim != dim_v else nn.Identity()
        self.cross_attn = nn.MultiheadAttention(dim_v, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim_v)
        self.conv_head = nn.Sequential(nn.Conv2d(dim_v, 256, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 1, 1))

    def forward(self, visual_feat: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        prompt = self.prompt_proj(prompt)
        (attn_out, _) = self.cross_attn(query=visual_feat, key=prompt, value=prompt)
        visual_feat = self.norm(visual_feat + attn_out)
        bt = visual_feat.size(0)
        s = self.spatial_size
        x = visual_feat.permute(0, 2, 1).contiguous().view(bt, -1, s, s)
        x = self.conv_head(x)
        x = F.interpolate(x, size=(self.mask_size, self.mask_size), mode='bilinear', align_corners=False)
        return x.squeeze(1)

class RefAVSBaselineModel(nn.Module):

    def __init__(self, pretrained_m2f: str | None='facebook/mask2former-swin-base-ade-semantic', dim_v: int=1024, num_heads: int=8, audio_dim: int=128, text_dim: int=768, mask_size: int=256, image_size: int=256, freeze_backbone: bool=False, prompt_dim: int=256, cache_mem_beta: float=1.0, **_):
        super().__init__()
        if not _HF_AVAILABLE:
            raise ImportError('transformers is required for RefAVSBaselineModel.\nInstall via: pip install transformers')
        loaded_pretrained = False
        if pretrained_m2f is not None:
            try:
                _m2f = Mask2FormerModel.from_pretrained(pretrained_m2f)
                loaded_pretrained = True
            except Exception as e:
                print(f"[RefAVSBaselineModel] Failed to load pretrained Mask2Former ('{pretrained_m2f}'): {e}. Falling back to random init.")
                _m2f = Mask2FormerModel(Mask2FormerConfig())
        else:
            _m2f = Mask2FormerModel(Mask2FormerConfig())
        self._swin: nn.Module = _m2f.pixel_level_module.encoder
        del _m2f
        if freeze_backbone and loaded_pretrained:
            for p in self._swin.parameters():
                p.requires_grad_(False)
        elif freeze_backbone and (not loaded_pretrained):
            print('[RefAVSBaselineModel] freeze_backbone=True but pretrained weights are unavailable. Keeping backbone trainable to avoid training collapse.')
        self.dim_v = dim_v
        self.cache_mem_beta = float(cache_mem_beta)
        spatial_size = image_size // 32
        self.spatial_size = spatial_size
        self.vis_proj = nn.Linear(1024, dim_v)
        self.audio_proj = nn.Sequential(nn.Linear(audio_dim, 2048), nn.ReLU(), nn.Linear(2048, dim_v))
        self.text_proj = nn.Sequential(nn.Linear(text_dim, 2048), nn.ReLU(), nn.Linear(2048, dim_v))
        self.prompt_proj = nn.Sequential(nn.Linear(dim_v, 2048), nn.ReLU(), nn.Linear(2048, prompt_dim))
        self.mha_A_T = nn.MultiheadAttention(dim_v, num_heads, batch_first=False)
        self.mha_V_T = nn.MultiheadAttention(dim_v, num_heads, batch_first=False)
        self.mha_mm = nn.MultiheadAttention(dim_v, num_heads, batch_first=False)
        self.tag_A = nn.Parameter(torch.zeros(1, 1, dim_v))
        self.tag_V = nn.Parameter(torch.ones(1, 1, dim_v))
        self.decoder = _CrossAttnDecoder(dim_v=dim_v, num_heads=num_heads, mask_size=mask_size, spatial_size=spatial_size, prompt_dim=prompt_dim)

    def _cached_memory(self, feat: torch.Tensor) -> torch.Tensor:
        beta = self.cache_mem_beta
        feat_beta = feat * (beta + 1)
        cum = torch.cumsum(feat, dim=0)
        idx = torch.arange(1, feat.size(0) + 1, device=feat.device, dtype=feat.dtype).view(-1, 1, 1)
        mean = cum / idx
        return feat_beta - mean

    def _encode_visual(self, frames: torch.Tensor) -> torch.Tensor:
        frozen = not next(self._swin.parameters()).requires_grad
        if frozen:
            with torch.no_grad():
                out = self._swin(pixel_values=frames)
        else:
            out = self._swin(pixel_values=frames)
        enc = self._to_tokens(out.feature_maps[-1])
        enc = self.vis_proj(enc)
        return enc

    @staticmethod
    def _to_tokens(enc: torch.Tensor) -> torch.Tensor:
        if enc.dim() == 3:
            return enc
        if enc.dim() == 4:
            if enc.size(-1) > enc.size(1):
                return enc.reshape(enc.size(0), -1, enc.size(-1))
            else:
                return enc.flatten(2).transpose(1, 2)
        return enc

    def forward(self, frames: torch.Tensor, audio_feat: torch.Tensor, text_feat: torch.Tensor, masks: torch.Tensor | None=None, **_) -> dict:
        (b, t, c, h, w) = frames.shape
        vis_enc = self._encode_visual(frames.view(b * t, c, h, w))
        s = vis_enc.size(1)
        feat_vis = vis_enc.view(b, t * s, self.dim_v)
        feat_aud = self.audio_proj(audio_feat)
        feat_txt = self.text_proj(text_feat)
        fused_AT = torch.cat([feat_aud, feat_txt], dim=1).permute(1, 0, 2)
        fused_VT = torch.cat([feat_vis, feat_txt], dim=1).permute(1, 0, 2)
        (fused_AT, _) = self.mha_A_T(fused_AT, fused_AT, fused_AT)
        (fused_VT, _) = self.mha_V_T(fused_VT, fused_VT, fused_VT)
        ta = feat_aud.size(1)
        tv = feat_vis.size(1)
        cue_A_seq = fused_AT[:ta]
        cue_T_from_A = fused_AT[ta:]
        cue_V_seq = fused_VT[:tv]
        cue_T_from_V = fused_VT[tv:]
        cue_V_diff_flat = self._cached_memory(cue_V_seq)
        cue_A_diff = self._cached_memory(cue_A_seq)
        cue_T = (feat_txt + cue_T_from_A.permute(1, 0, 2) + cue_T_from_V.permute(1, 0, 2)) / 3.0
        tag_A = self.tag_A.expand(1, b, self.dim_v)
        tag_V = self.tag_V.expand(1, b, self.dim_v)
        cue_V_bt = cue_V_diff_flat.permute(1, 0, 2).contiguous().view(b, t, s, self.dim_v)
        cue_A_for_prompt = cue_A_diff
        prompts = []
        for f in range(t):
            cue_V_f = cue_V_bt[:, f].permute(1, 0, 2)
            cue_T_perm = cue_T.permute(1, 0, 2)
            mm_seq = torch.cat([cue_A_for_prompt, tag_A, cue_V_f, tag_V, cue_T_perm], dim=0)
            (mm_out, _) = self.mha_mm(mm_seq, mm_seq, mm_seq)
            mm_out = self.prompt_proj(mm_out.permute(1, 0, 2))
            prompts.append(mm_out)
        prompts = torch.cat(prompts, dim=0)
        vis_enc_bt = vis_enc
        logits = self.decoder(vis_enc_bt, prompts)
        return {'logits': logits}

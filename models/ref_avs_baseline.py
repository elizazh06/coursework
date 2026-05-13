"""RefAVSBaselineModel — faithful adaptation of REFAVS_Model_Base
(https://github.com/GeWu-Lab/Ref-AVS/blob/main/models/avs_model.py)

Changes vs. the original:
  • Uses the standard HuggingFace `Mask2FormerModel` backbone instead of the
    custom local fork, making the model self-contained.
  • The segmentation head is a lightweight cross-attention decoder that
    conditions visual features on multimodal prompts and upsamples to
    mask_size × mask_size — analogous to the prompt-conditioned Mask2Former
    decoder used by the authors.
  • Returns a plain dict {"logits": Tensor[B*T, H, W]} compatible with
    the project's Trainer + SegmentationLoss.

Tensor shapes (all internal):
  dim_v = 1024  (matches Swin-Base encoder channels)
  B     = batch size
  T     = num_frames per clip
  T_a   = audio feature sequence length (pre-extracted, typically == T)
  T_t   = text feature sequence length (BERT tokens)
  S     = spatial tokens from visual encoder = (image_size/32)^2
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import Mask2FormerModel, Mask2FormerConfig
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False


# ---------------------------------------------------------------------------
# Segmentation head
# ---------------------------------------------------------------------------

class _CrossAttnDecoder(nn.Module):
    """
    Lightweight decoder that conditions spatial visual features on multimodal
    prompt embeddings via cross-attention, then upsamples to mask_size.

    Q = visual features  [B*T, S, dim_v]
    K = V = prompt        [B*T, L, dim_v]  (L = prompt seq len)
    output                [B*T, 1, mask_size, mask_size]
    """

    def __init__(self, dim_v: int, num_heads: int, mask_size: int, spatial_size: int):
        super().__init__()
        self.spatial_size = spatial_size
        self.mask_size = mask_size

        self.cross_attn = nn.MultiheadAttention(dim_v, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim_v)

        # project dim_v → 1 channel then upsample
        self.conv_head = nn.Sequential(
            nn.Conv2d(dim_v, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, visual_feat: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_feat: [B*T, S, dim_v]
            prompt:      [B*T, L, dim_v]
        Returns:
            logits: [B*T, mask_size, mask_size]  (raw, before sigmoid)
        """
        attn_out, _ = self.cross_attn(query=visual_feat, key=prompt, value=prompt)
        visual_feat = self.norm(visual_feat + attn_out)  # residual

        bt = visual_feat.size(0)
        s = self.spatial_size
        # reshape to spatial map: [B*T, dim_v, s, s]
        x = visual_feat.permute(0, 2, 1).contiguous().view(bt, -1, s, s)
        x = self.conv_head(x)   # [B*T, 1, s, s]
        x = F.interpolate(x, size=(self.mask_size, self.mask_size),
                          mode="bilinear", align_corners=False)
        return x.squeeze(1)   # [B*T, mask_size, mask_size]


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class RefAVSBaselineModel(nn.Module):
    """
    Audio-visual referring segmentation baseline for Ref-AVS-Bench.

    Follows the architecture of REFAVS_Model_Base:
      1. Encode frames with Mask2Former (Swin-Base) backbone.
      2. Project audio features [T_a, 128] and text features [T_t, 768]
         to dim_v via two-layer MLPs.
      3. Fuse audio+text and visual+text via separate MultiheadAttention blocks
         (mha_A_T, mha_V_T).
      4. Apply causal cached-memory differential encoding to audio and visual
         cue streams (temporal context).
      5. Construct per-frame multimodal prompts via mha_mm.
      6. Decode: cross-attend visual spatial features on per-frame prompts,
         upsample to mask_size × mask_size.

    Args:
        pretrained_m2f: HuggingFace model id for Mask2Former backbone.
            Set to None to initialise from scratch (useful for tests).
        dim_v: internal feature dimension (1024 for Swin-Base).
        num_heads: attention heads in all MHA modules.
        audio_dim: dimension of pre-extracted audio features (default 128).
        text_dim: dimension of pre-extracted text features (default 768).
        mask_size: output mask resolution (H = W = mask_size).
        image_size: spatial size of input frames (assumes square).
        freeze_backbone: if True, backbone weights are frozen.
    """

    def __init__(
        self,
        pretrained_m2f: str | None = "facebook/mask2former-swin-base-ade-semantic",
        dim_v: int = 1024,
        num_heads: int = 8,
        audio_dim: int = 128,
        text_dim: int = 768,
        mask_size: int = 256,
        image_size: int = 256,
        freeze_backbone: bool = False,
        **_,   # absorb leftover keys from config deep-merge (e.g. advance params)
    ):
        super().__init__()

        if not _HF_AVAILABLE:
            raise ImportError(
                "transformers is required for RefAVSBaselineModel.\n"
                "Install via: pip install transformers"
            )

        # ---- visual backbone ----
        if pretrained_m2f is not None:
            self.backbone = Mask2FormerModel.from_pretrained(pretrained_m2f)
        else:
            cfg = Mask2FormerConfig()
            self.backbone = Mask2FormerModel(cfg)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        self.dim_v = dim_v

        # Spatial token count at encoder last hidden state.
        # Swin-Base: output stride 32 → (image_size // 32)^2 tokens.
        spatial_size = image_size // 32
        self.spatial_size = spatial_size

        # Encoder channels → dim_v projection
        # Mask2FormerModel encoder_last_hidden_state channels: 1024 for Swin-Base
        self.vis_proj = nn.Linear(1024, dim_v)

        # ---- audio / text projections (identical to paper) ----
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, dim_v),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, dim_v),
        )
        # prompt: dim_v → 256 (following the paper's prompt_proj)
        self.prompt_proj = nn.Sequential(
            nn.Linear(dim_v, 2048),
            nn.ReLU(),
            nn.Linear(2048, dim_v),   # keep dim_v for decoder
        )

        # ---- fusion attention (paper: mha_A_T, mha_V_T, mha_mm) ----
        self.mha_A_T = nn.MultiheadAttention(dim_v, num_heads, batch_first=False)
        self.mha_V_T = nn.MultiheadAttention(dim_v, num_heads, batch_first=False)
        self.mha_mm = nn.MultiheadAttention(dim_v, num_heads, batch_first=False)

        # tag embeddings for audio / visual modality in the mm prompt
        self.tag_A = nn.Parameter(torch.zeros(1, 1, dim_v))
        self.tag_V = nn.Parameter(torch.ones(1, 1, dim_v))

        # ---- segmentation decoder ----
        self.decoder = _CrossAttnDecoder(dim_v, num_heads, mask_size, spatial_size)

    # ------------------------------------------------------------------
    # cached memory (paper: process_with_cached_memory)
    # ------------------------------------------------------------------

    @staticmethod
    def _cached_memory(feat: torch.Tensor) -> torch.Tensor:
        """
        Temporal differential encoding via a causal cumulative mean.

        feat: [T, B, dim_v]  (seq_len, batch, dim)
        out:  [T, B, dim_v]
        """
        beta = 1.0
        feat_beta = feat * (beta + 1)
        cum = torch.cumsum(feat, dim=0)
        idx = torch.arange(1, feat.size(0) + 1, device=feat.device,
                           dtype=feat.dtype).view(-1, 1, 1)
        mean = cum / idx
        return feat_beta - mean

    # ------------------------------------------------------------------
    # visual encoding
    # ------------------------------------------------------------------

    def _encode_visual(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode all B*T frames through the Mask2Former backbone.

        When frozen: torch.no_grad() skips storing activations for backprop
        (~2 GB saved for 10 frames) and all frames are processed in a single
        batch forward pass (10x faster than frame-by-frame).

        Args:
            frames: [B*T, C, H, W]
        Returns:
            enc: [B*T, S, dim_v]   (S = spatial_size^2)
        """
        frozen = not any(p.requires_grad for p in self.backbone.parameters())
        if frozen:
            with torch.no_grad():
                outputs = self.backbone(pixel_values=frames)
        else:
            outputs = self.backbone(pixel_values=frames)

        enc = self._to_tokens(outputs.encoder_last_hidden_state)
        enc = self.vis_proj(enc)   # [B*T, S, dim_v]
        return enc

    @staticmethod
    def _to_tokens(enc: torch.Tensor) -> torch.Tensor:
        """Flatten spatial dims to a token sequence → [B, S, C].

        Swin returns NHWC  [B, H, W, C].
        Standard CNN/ViT may return NCHW [B, C, H, W].
        """
        if enc.dim() == 4:
            # Detect format: for Swin-Base C=1024 >> typical spatial H,W (e.g. 8)
            # so last dim being the largest signals NHWC.
            if enc.size(-1) > enc.size(1):
                # NHWC [B, H, W, C] → [B, H*W, C]
                return enc.reshape(enc.size(0), -1, enc.size(-1))
            else:
                # NCHW [B, C, H, W] → [B, H*W, C]
                return enc.flatten(2).transpose(1, 2)
        return enc   # already [B, S, C]

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        frames: torch.Tensor,     # [B, T, C, H, W]
        audio_feat: torch.Tensor, # [B, T_a, 128]
        text_feat: torch.Tensor,  # [B, T_t, 768]
        masks: torch.Tensor | None = None,   # [B, T, H_m, W_m]  (unused here)
        **_,
    ) -> dict:
        b, t, c, h, w = frames.shape

        # ---- 1. Visual encoding ----
        vis_enc = self._encode_visual(frames.view(b * t, c, h, w))  # [B*T, S, dim_v]
        s = vis_enc.size(1)
        # reshape to [B, T*S, dim_v]
        feat_vis = vis_enc.view(b, t * s, self.dim_v)

        # ---- 2. Audio / text projection ----
        feat_aud = self.audio_proj(audio_feat)   # [B, T_a, dim_v]
        feat_txt = self.text_proj(text_feat)     # [B, T_t, dim_v]

        # ---- 3. Joint A+T and V+T fusion (paper section 3.3) ----
        # concatenate along seq dim, permute for nn.MultiheadAttention (L, B, D)
        fused_AT = torch.cat([feat_aud, feat_txt], dim=1).permute(1, 0, 2)   # [T_a+T_t, B, dim_v]
        fused_VT = torch.cat([feat_vis, feat_txt], dim=1).permute(1, 0, 2)   # [T*S+T_t, B, dim_v]

        fused_AT, _ = self.mha_A_T(fused_AT, fused_AT, fused_AT)
        fused_VT, _ = self.mha_V_T(fused_VT, fused_VT, fused_VT)

        # split back into audio/visual and text parts
        ta = feat_aud.size(1)
        tv = feat_vis.size(1)

        cue_A_seq = fused_AT[:ta]           # [T_a, B, dim_v]
        cue_T_from_A = fused_AT[ta:]        # [T_t, B, dim_v]
        cue_V_seq = fused_VT[:tv]           # [T*S, B, dim_v]
        cue_T_from_V = fused_VT[tv:]        # [T_t, B, dim_v]

        # ---- 4. Cached memory differential encoding ----
        # Reshape visual cues to [T, B*S, dim_v] for per-frame memory
        cue_V_seq_ts = cue_V_seq.view(t, s, b, self.dim_v).permute(0, 2, 1, 3)
        # cue_V_seq_ts: [T, B, S, dim_v] → process as [T, B*S, dim_v]
        cue_V_flat = cue_V_seq_ts.contiguous().view(t, b * s, self.dim_v)
        cue_V_diff = self._cached_memory(cue_V_flat)   # [T, B*S, dim_v]

        cue_A_diff = self._cached_memory(cue_A_seq)    # [T_a, B, dim_v]

        # Combined text cue [B, T_t, dim_v]
        cue_T = (feat_txt +
                 cue_T_from_A.permute(1, 0, 2) +
                 cue_T_from_V.permute(1, 0, 2)) / 3.0

        # ---- 5. Per-frame multimodal prompt (paper: mha_mm) ----
        # For each frame f, build: [A_diff[avg], tag_A, V_diff_f, tag_V, T_cue]
        cue_A_mean = cue_A_diff.mean(dim=0, keepdim=True)  # [1, B, dim_v]
        tag_A = self.tag_A.expand(1, b, self.dim_v)        # [1, B, dim_v]
        tag_V = self.tag_V.expand(1, b, self.dim_v)        # [1, B, dim_v]

        prompts = []
        for f in range(t):
            # Visual cue for frame f: [B, S, dim_v] → permute [S, B, dim_v]
            cue_V_f = cue_V_diff[f].view(b, s, self.dim_v).permute(1, 0, 2)  # [S, B, dim_v]
            cue_T_perm = cue_T.permute(1, 0, 2)   # [T_t, B, dim_v]

            mm_seq = torch.cat([cue_A_mean, tag_A, cue_V_f, tag_V, cue_T_perm], dim=0)
            mm_out, _ = self.mha_mm(mm_seq, mm_seq, mm_seq)
            mm_out = self.prompt_proj(mm_out.permute(1, 0, 2))  # [B, L, dim_v]
            prompts.append(mm_out)

        # prompts: list of T tensors [B, L, dim_v] → [B*T, L, dim_v]
        prompts = torch.cat(prompts, dim=0)   # [B*T, L, dim_v]

        # ---- 6. Decode: cross-attend visual tokens on prompts ----
        vis_enc_bt = vis_enc   # [B*T, S, dim_v]  (already computed above)
        logits = self.decoder(vis_enc_bt, prompts)   # [B*T, mask_size, mask_size]

        return {"logits": logits}

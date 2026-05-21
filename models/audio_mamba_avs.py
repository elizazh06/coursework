from __future__ import annotations
import torch
import torch.nn as nn
from models.mixture_of_mamba import RMSNorm
from models.ssm_avs_common import BaseSSMAVSModel, MambaSequenceBlock


class AudioMambaAVSModel(BaseSSMAVSModel):

    def __init__(self, d_model: int=768, n_audio_layers: int=2, n_fusion_layers: int=2, d_state: int=16, conv_kernel: int=4, expand: int=2, ff_mult: int=4, dropout: float=0.1, max_audio_tokens: int=10, max_text_tokens: int=25, mask_size: int=256, audio_dim: int=128, text_dim: int=768, pretrained_visual_model: str | None='facebook/mask2former-swin-base-ade-semantic', freeze_visual_backbone: bool=True, use_official_mamba_ssm: bool=True, hf_pretrained_mamba_model: str | None='state-spaces/mamba-130m-hf', pretrained_mamba_path: str | None=None, pretrained_mamba_prefix: str='', freeze_mamba: bool=False, **_):
        super().__init__(d_model=d_model, audio_dim=audio_dim, text_dim=text_dim, max_audio_tokens=max_audio_tokens, max_text_tokens=max_text_tokens, mask_size=mask_size, pretrained_visual_model=pretrained_visual_model, freeze_visual_backbone=freeze_visual_backbone, dropout=dropout)
        self.audio_blocks = nn.ModuleList([MambaSequenceBlock(d_model=d_model, d_state=d_state, conv_kernel=conv_kernel, expand=expand, ff_mult=ff_mult, dropout=dropout, use_official_mamba_ssm=use_official_mamba_ssm) for _ in range(n_audio_layers)])
        self.audio_blocks_norm = RMSNorm(d_model)
        self.fusion_blocks = nn.ModuleList([MambaSequenceBlock(d_model=d_model, d_state=d_state, conv_kernel=conv_kernel, expand=expand, ff_mult=ff_mult, dropout=dropout, use_official_mamba_ssm=use_official_mamba_ssm) for _ in range(n_fusion_layers)])
        self.fusion_blocks_norm = RMSNorm(d_model)
        self.modality_embed = nn.Parameter(torch.zeros(1, 4, d_model))
        self.query_embed = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 10 + max_audio_tokens + max_text_tokens + 10, d_model))
        nn.init.trunc_normal_(self.modality_embed, std=0.02)
        nn.init.trunc_normal_(self.query_embed, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if hf_pretrained_mamba_model:
            self._load_pretrained_mamba_from_hf(hf_pretrained_mamba_model, ('audio_blocks', 'fusion_blocks'))
        if pretrained_mamba_path:
            self._load_pretrained_mamba_checkpoint(pretrained_mamba_path, pretrained_mamba_prefix, ('audio_blocks.', 'audio_blocks_norm.', 'fusion_blocks.', 'fusion_blocks_norm.'))
        if freeze_mamba:
            for module in (self.audio_blocks, self.audio_blocks_norm, self.fusion_blocks, self.fusion_blocks_norm):
                for p in module.parameters():
                    p.requires_grad = False

    def forward(self, frames: torch.Tensor, audio_feat: torch.Tensor, text_feat: torch.Tensor, masks: torch.Tensor | None=None, **_) -> dict:
        del masks
        spatial, video_tokens, audio_tokens, text_tokens = self._project_inputs(frames, audio_feat, text_feat)
        for block in self.audio_blocks:
            audio_tokens = block(audio_tokens)
        audio_tokens = self.audio_blocks_norm(audio_tokens)
        b, t = video_tokens.shape[:2]
        queries = self.query_embed.expand(b, t, -1)
        x = torch.cat([video_tokens + self.modality_embed[:, 0:1], audio_tokens + self.modality_embed[:, 1:2], text_tokens + self.modality_embed[:, 2:3], queries + self.modality_embed[:, 3:4]], dim=1)
        x = self.dropout(x + self.pos_embed[:, :x.size(1)])
        for block in self.fusion_blocks:
            x = block(x)
        x = self.fusion_blocks_norm(x)
        q_start = t + audio_tokens.size(1) + text_tokens.size(1)
        queries = x[:, q_start:q_start + t]
        return {'logits': self._decode(queries, spatial)}

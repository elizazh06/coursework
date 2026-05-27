from __future__ import annotations
import torch
import torch.nn as nn
from models.mixture_of_mamba import BaseAVSceneModel, MambaSequenceBlock, RMSNorm


class VideoAudioMambaModel(BaseAVSceneModel):

    def __init__(self, d_model=768, n_visual_layers=2, n_audio_layers=2, n_fusion_layers=2, d_state=16, conv_kernel=4, expand=2, ff_mult=4, dropout=0.1, max_audio_tokens=10, max_text_tokens=25, mask_size=256, audio_dim=128, text_dim=768, pretrained_visual_model='facebook/mask2former-swin-base-ade-semantic', freeze_visual_backbone=True, use_official_mamba_ssm=True, hf_pretrained_mamba_model='state-spaces/mamba-130m-hf', load_pretrained_mamba=True, pretrained_mamba_path=None, pretrained_mamba_prefix='', freeze_mamba=False, **_):
        super().__init__(d_model=d_model, audio_dim=audio_dim, text_dim=text_dim, max_audio_tokens=max_audio_tokens, max_text_tokens=max_text_tokens, mask_size=mask_size, pretrained_visual_model=pretrained_visual_model, freeze_visual_backbone=freeze_visual_backbone, dropout=dropout)
        self.visual_blocks = nn.ModuleList([MambaSequenceBlock(d_model=d_model, d_state=d_state, conv_kernel=conv_kernel, expand=expand, ff_mult=ff_mult, dropout=dropout, use_official_mamba_ssm=use_official_mamba_ssm) for _ in range(n_visual_layers)])
        self.visual_blocks_norm = RMSNorm(d_model)
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
        if load_pretrained_mamba and hf_pretrained_mamba_model:
            self._load_pretrained_mamba_from_hf(hf_pretrained_mamba_model, ('visual_blocks', 'audio_blocks', 'fusion_blocks'))
        if pretrained_mamba_path:
            self._load_pretrained_mamba_checkpoint(pretrained_mamba_path, pretrained_mamba_prefix, ('visual_blocks.', 'visual_blocks_norm.', 'audio_blocks.', 'audio_blocks_norm.', 'fusion_blocks.', 'fusion_blocks_norm.'))
        if freeze_mamba:
            for module in (self.visual_blocks, self.visual_blocks_norm, self.audio_blocks, self.audio_blocks_norm, self.fusion_blocks, self.fusion_blocks_norm):
                for p in module.parameters():
                    p.requires_grad = False

    def forward(self, frames, audio_feat, text_feat=None, masks=None, **_):
        del masks
        spatial, _, audio_tokens, text_tokens = self._project_inputs(frames, audio_feat, text_feat)
        b, t, d, h, w = spatial.shape
        visual_seq = spatial.permute(0, 1, 3, 4, 2).contiguous().view(b * t, h * w, d)
        for block in self.visual_blocks:
            visual_seq = block(visual_seq)
        visual_seq = self.visual_blocks_norm(visual_seq)
        spatial = visual_seq.view(b, t, h, w, d).permute(0, 1, 4, 2, 3).contiguous()
        video_tokens = spatial.mean(dim=(-1, -2))
        for block in self.audio_blocks:
            audio_tokens = block(audio_tokens)
        audio_tokens = self.audio_blocks_norm(audio_tokens)
        queries = self.query_embed.expand(b, t, -1)
        x = torch.cat([video_tokens + self.modality_embed[:, 0:1], audio_tokens + self.modality_embed[:, 1:2], text_tokens + self.modality_embed[:, 2:3], queries + self.modality_embed[:, 3:4]], dim=1)
        x = self.dropout(x + self.pos_embed[:, :x.size(1)])
        for block in self.fusion_blocks:
            x = block(x)
        x = self.fusion_blocks_norm(x)
        q_start = t + audio_tokens.size(1) + text_tokens.size(1)
        queries = x[:, q_start:q_start + t]
        return {'logits': self._decode(queries, spatial)}

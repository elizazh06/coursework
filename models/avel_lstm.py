from __future__ import annotations
import torch
import torch.nn as nn
from models.mixture_of_mamba import BaseAVSceneModel


class AVELLSTMModel(BaseAVSceneModel):

    def __init__(self, d_model=512, lstm_hidden=256, n_lstm_layers=2, dropout=0.1, max_audio_tokens=10, max_text_tokens=25, mask_size=256, audio_dim=128, text_dim=768, pretrained_visual_model='facebook/mask2former-swin-base-ade-semantic', freeze_visual_backbone=True, freeze_audio_adapter=False, **_):
        super().__init__(d_model=d_model, audio_dim=audio_dim, text_dim=text_dim, max_audio_tokens=max_audio_tokens, max_text_tokens=max_text_tokens, mask_size=mask_size, pretrained_visual_model=pretrained_visual_model, freeze_visual_backbone=freeze_visual_backbone, dropout=dropout)
        self.audio_context = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model))
        self.text_context = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model))
        self.av_gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.input_proj = nn.Linear(d_model * 4, d_model)
        self.temporal_lstm = nn.LSTM(input_size=d_model, hidden_size=lstm_hidden, num_layers=n_lstm_layers, batch_first=True, dropout=dropout if n_lstm_layers > 1 else 0.0, bidirectional=True)
        self.query_proj = nn.Sequential(nn.LayerNorm(lstm_hidden * 2), nn.Linear(lstm_hidden * 2, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model))
        if freeze_audio_adapter:
            for module in (self.audio_proj, self.audio_context):
                for p in module.parameters():
                    p.requires_grad = False

    def forward(self, frames, audio_feat, text_feat=None, masks=None, **_):
        del masks
        spatial, video_tokens, audio_tokens, text_tokens = self._project_inputs(frames, audio_feat, text_feat)
        t = video_tokens.size(1)
        if audio_tokens.size(1) != t:
            audio_tokens = torch.nn.functional.interpolate(audio_tokens.transpose(1, 2), size=t, mode='linear', align_corners=False).transpose(1, 2)
        audio_context = self.audio_context(audio_tokens)
        text_context = self.text_context(text_tokens.mean(dim=1, keepdim=True)).expand(-1, t, -1)
        gate = self.av_gate(torch.cat([video_tokens, audio_context], dim=-1))
        attended_video = video_tokens * gate
        fused = torch.cat([attended_video, audio_context, text_context, attended_video * audio_context], dim=-1)
        fused = self.input_proj(fused)
        temporal, _ = self.temporal_lstm(fused)
        queries = self.query_proj(temporal)
        return {'logits': self._decode(queries, spatial)}

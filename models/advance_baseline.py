import torch
import torch.nn as nn
import torchvision.models as models


class ADVANCEBaselineModel(nn.Module):
    """
    Simple audio-visual baseline for ADVANCE classification.
    Expects:
      - video: [B, T, C, H, W] (T can be 1)
      - audio: [B, Tm, F]
      - question: unused, kept for pipeline compatibility
    """

    def __init__(self, num_classes=13, hidden_dim=256, dropout=0.2, use_pretrained_backbone=True):
        super().__init__()
        weights = None
        if use_pretrained_backbone:
            try:
                weights = models.ResNet18_Weights.DEFAULT
            except AttributeError:
                weights = None
        self.visual_backbone = models.resnet18(weights=weights)
        self.visual_backbone.fc = nn.Identity()
        self.visual_proj = nn.Linear(512, hidden_dim)

        self.audio_encoder = nn.Sequential(
            nn.Conv1d(128, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, video, audio, question=None):
        b, t, c, h, w = video.shape
        vf = self.visual_backbone(video.view(b * t, c, h, w)).view(b, t, -1).mean(dim=1)
        vf = self.visual_proj(vf)

        af = self.audio_encoder(audio.transpose(1, 2)).mean(dim=2)
        return self.fusion(torch.cat([vf, af], dim=1))

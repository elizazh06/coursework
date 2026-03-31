import torch
import torch.nn as nn
import torchvision.models as models


class AVQABaselineModel(nn.Module):
    def __init__(
        self,
        vocab_size=5000,
        embed_dim=256,
        hidden_dim=512,
        num_classes=10
    ):
        super().__init__()
        
        resnet = models.resnet18(pretrained=True)
        self.video_encoder = nn.Sequential(*list(resnet.children())[:-1])  # remove fc
        self.video_fc = nn.Linear(512, hidden_dim)

        self.audio_encoder = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        self.fusion = nn.Linear(hidden_dim * 3, hidden_dim)

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, video, audio, question):
        """
        video: (B, T, C, H, W)
        audio: (B, T, F)
        question: (B, L)
        """

        B, T, C, H, W = video.shape

        video = video.view(B * T, C, H, W)
        v_feat = self.video_encoder(video)
        v_feat = v_feat.view(B, T, -1)
        v_feat = self.video_fc(v_feat)
        v_feat = v_feat.mean(dim=1)

        a_feat = audio.transpose(1, 2)
        a_feat = self.audio_encoder(a_feat)
        a_feat = a_feat.mean(dim=2)

        q_emb = self.embedding(question)
        _, (h, _) = self.lstm(q_emb)
        q_feat = h[-1]

        fused = torch.cat([v_feat, a_feat, q_feat], dim=1)
        fused = self.fusion(fused)

        return self.classifier(fused)
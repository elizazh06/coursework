import torch
import torch.nn as nn
import torch.nn.functional as F


class QuestionEncoder(nn.Module):
    def __init__(self, vocab_size, word_embed_size=256, embed_size=512, num_layers=1, hidden_size=512):
        super().__init__()
        self.word2vec = nn.Embedding(vocab_size, word_embed_size)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(2 * num_layers * hidden_size, embed_size)

    def forward(self, question):
        qst_vec = self.tanh(self.word2vec(question))
        self.lstm.flatten_parameters()
        _, (hidden, cell) = self.lstm(qst_vec)
        qst_feature = torch.cat((hidden, cell), dim=2)
        qst_feature = qst_feature.transpose(0, 1).reshape(question.size(0), -1)
        qst_feature = self.fc(self.tanh(qst_feature))
        return qst_feature


class MusicAVQABaselineModel(nn.Module):
    """
    Adapted from the official MUSIC-AVQA baseline:
    - audio: [B, T, 128] (VGGish-like feature)
    - video: [B, T, 512, 14, 14] (ResNet-18 14x14 feature map)
    - question: [B, L]
    """

    def __init__(
        self,
        vocab_size=5000,
        hidden_dim=512,
        num_classes=42,
        dropout=0.1,
        num_heads=4,
    ):
        super().__init__()
        self.fc_a1 = nn.Linear(128, hidden_dim)
        self.fc_a2 = nn.Linear(hidden_dim, hidden_dim)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_fusion = nn.Linear(hidden_dim * 2, hidden_dim)

        self.question_encoder = QuestionEncoder(
            vocab_size=vocab_size,
            word_embed_size=hidden_dim,
            embed_size=hidden_dim,
            num_layers=1,
            hidden_size=hidden_dim,
        )

        self.attn_a = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=False)
        self.attn_v = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=False)

        self.linear11 = nn.Linear(hidden_dim, hidden_dim)
        self.linear12 = nn.Linear(hidden_dim, hidden_dim)
        self.linear21 = nn.Linear(hidden_dim, hidden_dim)
        self.linear22 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.tanh = nn.Tanh()
        self.fc_ans = nn.Linear(hidden_dim, num_classes)

    def forward(self, video, audio, question):
        qst_feature = self.question_encoder(question)
        xq = qst_feature.unsqueeze(0)

        audio_feat = F.relu(self.fc_a1(audio))
        audio_feat = self.fc_a2(audio_feat)

        b, t, c, h, w = video.size()
        visual_feat = self.avgpool(video.view(b * t, c, h, w))
        visual_feat = visual_feat.squeeze(-1).squeeze(-1).view(b, t, c)

        visual_feat_grd = visual_feat.permute(1, 0, 2)
        visual_feat_att = self.attn_v(xq, visual_feat_grd, visual_feat_grd)[0].squeeze(0)
        src_v = self.linear12(self.dropout1(F.relu(self.linear11(visual_feat_att))))
        visual_feat_att = self.norm1(visual_feat_att + self.dropout2(src_v))

        audio_feat = audio_feat.permute(1, 0, 2)
        audio_feat_att = self.attn_a(xq, audio_feat, audio_feat)[0].squeeze(0)
        src_a = self.linear22(self.dropout3(F.relu(self.linear21(audio_feat_att))))
        audio_feat_att = self.norm2(audio_feat_att + self.dropout4(src_a))

        feat = self.tanh(self.fc_fusion(torch.cat((audio_feat_att, visual_feat_att), dim=-1)))
        combined_feature = self.tanh(feat * qst_feature)
        return self.fc_ans(combined_feature)

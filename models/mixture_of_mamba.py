import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


class SelectiveStateSpaceMixer(nn.Module):
    """
    Lightweight Mamba-style selective state-space mixer.
    """

    def __init__(self, d_model, d_state=64, conv_kernel=3, expand=2):
        super().__init__()
        inner_dim = d_model * expand
        self.in_proj = nn.Linear(d_model, inner_dim * 2)
        self.dw_conv = nn.Conv1d(
            inner_dim,
            inner_dim,
            kernel_size=conv_kernel,
            padding=conv_kernel - 1,
            groups=inner_dim,
        )
        self.dt_proj = nn.Linear(inner_dim, d_state)
        self.B_proj = nn.Linear(inner_dim, d_state)
        self.C_proj = nn.Linear(inner_dim, d_state)
        self.state_to_inner = nn.Linear(d_state, inner_dim)
        self.D = nn.Parameter(torch.ones(inner_dim))
        self.out_proj = nn.Linear(inner_dim, d_model)
        self.act = nn.SiLU()

    def forward(self, x):
        bsz, seq_len, _ = x.shape
        xz = self.in_proj(x)
        x_main, gate = xz.chunk(2, dim=-1)

        x_main = x_main.transpose(1, 2)
        x_main = self.dw_conv(x_main)[..., :seq_len]
        x_main = x_main.transpose(1, 2)
        x_main = self.act(x_main)

        dt = torch.sigmoid(self.dt_proj(x_main))
        B = self.B_proj(x_main)
        C = self.C_proj(x_main)

        state = x.new_zeros(bsz, B.size(-1))
        outputs = []
        for t in range(seq_len):
            state = (1.0 - dt[:, t]) * state + dt[:, t] * B[:, t]
            yt = C[:, t] * state
            outputs.append(yt)
        y = torch.stack(outputs, dim=1)

        # Broadcast low-rank state output back to channel space.
        y = F.layer_norm(y, (y.size(-1),))
        y = self.state_to_inner(y)
        y = y + self.D * x_main
        y = y * torch.sigmoid(gate)
        return self.out_proj(y)


class MoEFeedForward(nn.Module):
    def __init__(self, d_model, ff_mult=4, num_experts=4, top_k=2, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        hidden = d_model * ff_mult
        self.gate = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, d_model),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x):
        # x: [B, T, D]
        logits = self.gate(x)
        top_vals, top_idx = torch.topk(logits, k=self.top_k, dim=-1)
        top_probs = torch.softmax(top_vals, dim=-1)

        out = torch.zeros_like(x)
        for expert_id, expert in enumerate(self.experts):
            expert_out = expert(x)
            match = (top_idx == expert_id).float()
            weight = (top_probs * match).sum(dim=-1, keepdim=True)
            out = out + expert_out * weight
        return out


class MixtureOfMambaBlock(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=64,
        conv_kernel=3,
        expand=2,
        ff_mult=4,
        num_experts=4,
        top_k=2,
        dropout=0.1,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.mixer = SelectiveStateSpaceMixer(
            d_model=d_model,
            d_state=d_state,
            conv_kernel=conv_kernel,
            expand=expand,
        )
        self.norm2 = RMSNorm(d_model)
        self.moe = MoEFeedForward(
            d_model=d_model,
            ff_mult=ff_mult,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.mixer(self.norm1(x)))
        x = x + self.dropout(self.moe(self.norm2(x)))
        return x


class MixtureOfMambaModel(nn.Module):
    """
    Universal AV model with Mamba-style sequence mixer + MoE FFN.
    Works with both:
      - raw video frames [B, T, 3, H, W]
      - pre-extracted video tensors [B, T, C, H, W] (e.g., 512x14x14)
    Audio is expected as [B, T, F], question as [B, L] (optional).
    """

    def __init__(
        self,
        num_classes=13,
        d_model=256,
        hidden_dim=None,
        n_layers=4,
        d_state=64,
        conv_kernel=3,
        expand=2,
        ff_mult=4,
        num_experts=4,
        top_k=2,
        dropout=0.1,
        vocab_size=5000,
        max_video_tokens=32,
        max_audio_tokens=64,
        **_,
    ):
        super().__init__()
        if hidden_dim is not None:
            d_model = int(hidden_dim)
        self.max_video_tokens = max_video_tokens
        self.max_audio_tokens = max_audio_tokens

        self.image_stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.video_proj = nn.LazyLinear(d_model)
        self.audio_proj = nn.LazyLinear(d_model)
        self.question_embedding = nn.Embedding(vocab_size, d_model)
        self.question_proj = nn.Linear(d_model, d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.modality_embed = nn.Parameter(torch.zeros(1, 4, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + max_video_tokens + max_audio_tokens + 1, d_model))

        self.blocks = nn.ModuleList(
            [
                MixtureOfMambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    conv_kernel=conv_kernel,
                    expand=expand,
                    ff_mult=ff_mult,
                    num_experts=num_experts,
                    top_k=top_k,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.modality_embed, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _encode_video(self, video):
        bsz, timesteps = video.shape[:2]
        if video.dim() != 5:
            raise ValueError(f"Expected video shape [B, T, C, H, W], got {tuple(video.shape)}")

        c = video.size(2)
        if c == 3:
            frames = video.view(bsz * timesteps, c, video.size(3), video.size(4))
            feat = self.image_stem(frames).flatten(1)
            feat = feat.view(bsz, timesteps, -1)
        else:
            feat = video.mean(dim=(-1, -2))

        if feat.size(1) > self.max_video_tokens:
            feat = feat[:, : self.max_video_tokens]
        return self.video_proj(feat)

    def _encode_audio(self, audio):
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        if audio.dim() != 3:
            raise ValueError(f"Expected audio shape [B, T, F], got {tuple(audio.shape)}")
        if audio.size(1) > self.max_audio_tokens:
            audio = audio[:, : self.max_audio_tokens]
        return self.audio_proj(audio)

    def _encode_question(self, question, batch_size, device):
        if question is None:
            return torch.zeros(batch_size, 1, self.question_proj.out_features, device=device)
        if question.dim() == 1:
            question = question.unsqueeze(1)
        if question.dim() != 2:
            question = question.view(batch_size, -1)
        q = self.question_embedding(question.long().clamp(min=0))
        q = self.question_proj(q.mean(dim=1, keepdim=True))
        return q

    def forward(self, video, audio, question=None):
        bsz = video.size(0)
        device = video.device

        v_tokens = self._encode_video(video)
        a_tokens = self._encode_audio(audio)
        q_token = self._encode_question(question, bsz, device)
        cls = self.cls_token.expand(bsz, -1, -1)

        v_tokens = v_tokens + self.modality_embed[:, 0:1]
        a_tokens = a_tokens + self.modality_embed[:, 1:2]
        q_token = q_token + self.modality_embed[:, 2:3]
        cls = cls + self.modality_embed[:, 3:4]

        x = torch.cat([cls, v_tokens, a_tokens, q_token], dim=1)
        x = x + self.pos_embed[:, : x.size(1)]
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x[:, 0])
        return self.head(x)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        return x + self.pos_embedding[:, :x.size(1), :]


class GRUTransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        gru_layers=2,
        tf_layers=2,
        nhead=4,
        dropout=0.3,
        max_len=2048,
        ff_mult=4,
        bidirectional_gru=True,
        use_attention_pool=True,
    ):
        super().__init__()
        if hidden_dim % nhead != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by nhead ({nhead})"
            )

        # Input projection and positional encoding
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len)
        self.input_dropout = nn.Dropout(dropout)

        # Transformer first
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * ff_mult,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=tf_layers)

        # GRU second
        self.bidirectional_gru = bool(bidirectional_gru)
        gru_out_dim = hidden_dim * (2 if self.bidirectional_gru else 1)
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0,
            bidirectional=self.bidirectional_gru,
        )
        self.gru_proj = (
            nn.Identity()
            if gru_out_dim == hidden_dim
            else nn.Linear(gru_out_dim, hidden_dim)
        )

        # Two residual blocks with learnable layer scales
        self.residual1_norm = nn.LayerNorm(hidden_dim)
        self.residual2_norm = nn.LayerNorm(hidden_dim)
        self.residual1_scale = nn.Parameter(torch.ones(1, 1, hidden_dim))
        self.residual2_scale = nn.Parameter(torch.ones(1, 1, hidden_dim))

        # Sequence pooling (attention pooling is usually stronger than last-step only)
        self.use_attention_pool = bool(use_attention_pool)
        if self.use_attention_pool:
            self.pool_query = nn.Linear(hidden_dim, 1)
        else:
            self.pool_query = None

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, lengths=None):
        # ---- Input projection ----
        h = self.input_proj(x)
        h = self.input_norm(h)
        h = self.pos_encoder(h)
        h = self.input_dropout(h)

        # ---- Mask ----
        if lengths is not None:
            max_len = h.size(1)
            mask = torch.arange(max_len, device=x.device)[None, :] >= lengths[:, None]
        else:
            mask = None

        # ---- Transformer (first) ----
        trans_out = self.transformer(h, src_key_padding_mask=mask)
        h = self.residual1_norm((self.residual1_scale * trans_out) + h)  # residual block #1

        # ---- GRU (second) ----
        if lengths is not None:
            packed = pack_padded_sequence(
                h, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.gru(packed)
            gru_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        else:
            gru_out, _ = self.gru(h)
        gru_out = self.gru_proj(gru_out)
        out = self.residual2_norm((self.residual2_scale * gru_out) + h)  # residual block #2

        # ---- Pooling ----
        if self.use_attention_pool:
            score = self.pool_query(out).squeeze(-1)  # [B, T]
            if lengths is not None:
                t = out.size(1)
                pool_mask = torch.arange(t, device=x.device)[None, :] >= lengths[:, None]
                score = score.masked_fill(pool_mask, float("-inf"))
            weights = torch.softmax(score, dim=1).unsqueeze(-1)  # [B, T, 1]
            out = (weights * out).sum(dim=1)  # [B, H]
        else:
            if lengths is not None:
                idx = (lengths - 1).clamp(min=0).to(x.device)
                out = out[torch.arange(out.size(0), device=x.device), idx]
            else:
                out = out[:, -1]

        return self.fc(out)


class LabelSmoothingBCE(nn.Module):
    def __init__(self, smoothing=0.05, pos_weight=None):
        super().__init__()
        self.smoothing = smoothing

        if pos_weight is not None:
            self.register_buffer(
                "pos_weight",
                torch.as_tensor(pos_weight, dtype=torch.float32),
            )
        else:
            self.pos_weight = None

    def forward(self, logits, targets):
        targets = targets.float()
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing

        pw = None
        if self.pos_weight is not None:
            pw = self.pos_weight.to(device=logits.device, dtype=logits.dtype)

        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)
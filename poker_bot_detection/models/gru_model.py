import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.residual_proj = nn.Linear(input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, lengths=None):
        if lengths is not None:
            packed = pack_padded_sequence(
                x,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_out, _ = self.gru(packed)
            gru_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        else:
            gru_out, _ = self.gru(x)

        residual = self.residual_proj(x)

        out = self.layer_norm(gru_out + residual)
        if lengths is not None:
            # Select the last valid timestep for each sequence.
            idx = (lengths - 1).clamp(min=0).to(x.device)
            out = out[torch.arange(out.size(0), device=x.device), idx, :]
        else:
            out = out[:, -1, :]

        return self.fc(out)


class LabelSmoothingBCE(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        targets = targets.float()
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(logits, targets)
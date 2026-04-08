import statistics
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import json
import gzip
from pathlib import Path

from .features import encode_chunk


class PokerDataset(Dataset):
    def __init__(
        self,
        file_path,
        split=None,
        feature_mean=None,
        feature_std=None,
        norm_eps=None,
    ):
        if str(file_path).endswith(".gz"):
            with gzip.open(file_path, "rt", encoding="utf-8") as f:
                self.data = json.load(f)
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)

        # Public benchmark payload format:
        # { ..., "labeled_chunks": [ {"hands": [...], "is_bot": bool, ...}, ... ] }
        # Keep backward compatibility with the older "chunks" format.
        self.labeled_chunks = self.data.get("labeled_chunks", self.data.get("chunks", []))

        # Use benchmark-provided deterministic split when available.
        if split is not None:
            self.labeled_chunks = [
                chunk
                for chunk in self.labeled_chunks
                if isinstance(chunk, dict) and chunk.get("split") == split
            ]

        self._feature_mean = (
            feature_mean.clone().detach().float().cpu()
            if feature_mean is not None
            else None
        )
        self._feature_std = (
            feature_std.clone().detach().float().cpu()
            if feature_std is not None
            else None
        )
        self._norm_eps = float(norm_eps) if norm_eps is not None else None

    def __len__(self):
        return len(self.labeled_chunks)

    def __getitem__(self, idx):
        chunk_entry = self.labeled_chunks[idx]
        if isinstance(chunk_entry, dict):
            chunks = chunk_entry.get("hands", [])
            label = 1.0 if chunk_entry.get("is_bot", False) else 0.0
        else:
            # Backward compatibility with old list-based entries.
            chunks = chunk_entry
            label = 1.0 if chunks and chunks[0].get("label") == "bot" else 0.0

        sequence = []
        for chunk in chunks:
            sequence.append(encode_chunk(chunk))
        if not sequence:
            # Guard against malformed empty chunks.
            sequence.append(encode_chunk({}))

        x = torch.tensor(sequence, dtype=torch.float32)

        if self._feature_mean is not None and self._feature_std is not None:
            eps = self._norm_eps if self._norm_eps is not None else 1e-8
            x = (x - self._feature_mean) / (self._feature_std + eps)

        y = torch.tensor([label], dtype=torch.float32)

        return x, y


def _chunk_is_bot_and_hands(chunk_entry):
    """Return (is_bot: bool, num_hands: int) for one labeled chunk entry."""
    if isinstance(chunk_entry, dict):
        hands = chunk_entry.get("hands", [])
        is_bot = bool(chunk_entry.get("is_bot", False))
    else:
        hands = chunk_entry
        is_bot = bool(hands and hands[0].get("label") == "bot")
    n_hands = len(hands) if isinstance(hands, list) else 0
    return is_bot, n_hands


def apply_feature_normalization(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, eps: float) -> torch.Tensor:
    """Apply train-fitted standardization to a hand sequence tensor [seq_len, dim] or batch."""
    m = mean.to(device=x.device, dtype=x.dtype)
    s = std.to(device=x.device, dtype=x.dtype)
    return (x - m) / (s + eps)


def fit_feature_normalization(dataset: PokerDataset, input_dim: int, eps: float = 1e-8):
    """
    Population mean and std per feature dimension over all *hand* vectors in the dataset.
    Fit on train split only; pass resulting tensors to val PokerDataset.
    """
    sum_vec = torch.zeros(input_dim, dtype=torch.float64)
    sumsq_vec = torch.zeros(input_dim, dtype=torch.float64)
    n_rows = 0
    for i in range(len(dataset)):
        x, _ = dataset[i]
        if x.numel() == 0:
            continue
        if x.shape[-1] != input_dim:
            raise ValueError(
                f"Feature dim {x.shape[-1]} != config INPUT_DIM {input_dim}; "
                "update INPUT_DIM to match encode_hand() output size."
            )
        flat = x.reshape(-1, input_dim).double()
        sum_vec += flat.sum(dim=0)
        sumsq_vec += (flat * flat).sum(dim=0)
        n_rows += flat.size(0)

    if n_rows == 0:
        mean = torch.zeros(input_dim, dtype=torch.float32)
        std = torch.ones(input_dim, dtype=torch.float32)
        return mean, std

    mean = (sum_vec / n_rows).float()
    var = (sumsq_vec / n_rows) - (sum_vec / n_rows) ** 2
    std = torch.sqrt(torch.clamp(var.float(), min=0.0))
    std = torch.where(std < eps, torch.ones_like(std), std)
    return mean, std


def save_feature_norm(path, mean: torch.Tensor, std: torch.Tensor, input_dim: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mean": mean.cpu(),
        "std": std.cpu(),
        "input_dim": int(input_dim),
    }
    torch.save(payload, path)


def load_feature_norm(path):
    payload = torch.load(path, map_location="cpu")
    return payload["mean"], payload["std"], payload.get("input_dim")


def compute_balance_stats(dataset: PokerDataset) -> dict:
    """
    Aggregate stats for a split: chunk counts, bot/human counts, hands-per-chunk stats.
    """
    labeled = dataset.labeled_chunks
    n = len(labeled)
    bot_count = 0
    human_count = 0
    hand_counts = []
    for entry in labeled:
        is_bot, n_hands = _chunk_is_bot_and_hands(entry)
        if is_bot:
            bot_count += 1
        else:
            human_count += 1
        hand_counts.append(n_hands)

    if hand_counts:
        hands_stats = {
            "avg": float(statistics.mean(hand_counts)),
            "min": int(min(hand_counts)),
            "max": int(max(hand_counts)),
        }
    else:
        hands_stats = {"avg": 0.0, "min": 0, "max": 0}

    maj = max(bot_count, human_count, 1)
    minority_ratio = min(bot_count, human_count) / maj

    return {
        "num_chunks": n,
        "bot_count": bot_count,
        "human_count": human_count,
        "minority_ratio": minority_ratio,
        "hands_per_chunk": hands_stats,
    }


def train_sample_weights(dataset: PokerDataset) -> torch.Tensor:
    """
    Per-index sampling weights inversely proportional to class frequency (for WeightedRandomSampler).
    """
    stats = compute_balance_stats(dataset)
    n_bot = stats["bot_count"]
    n_human = stats["human_count"]
    if n_bot == 0 or n_human == 0:
        return torch.ones(len(dataset), dtype=torch.double)

    w_bot = 1.0 / float(n_bot)
    w_human = 1.0 / float(n_human)
    weights = []
    for entry in dataset.labeled_chunks:
        is_bot, _ = _chunk_is_bot_and_hands(entry)
        weights.append(w_bot if is_bot else w_human)
    return torch.tensor(weights, dtype=torch.double)


def poker_collate_fn(batch):
    """
    Pad variable-length chunk sequences for GRU input.
    Returns:
        x_padded: [batch_size, max_seq_len, input_dim]
        lengths: [batch_size]
        y: [batch_size, 1]
    """
    sequences, labels = zip(*batch)
    lengths = torch.tensor([seq.size(0) for seq in sequences], dtype=torch.long)
    x_padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    y = torch.stack(labels, dim=0)
    return x_padded, lengths, y
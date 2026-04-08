import statistics
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import json
import gzip
from utils.features import encode_chunk

class PokerDataset(Dataset):
    def __init__(self, file_path, split=None):
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
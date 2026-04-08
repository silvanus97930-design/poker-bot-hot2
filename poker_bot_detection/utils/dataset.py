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
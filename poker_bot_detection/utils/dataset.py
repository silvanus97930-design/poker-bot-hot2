import torch
from torch.utils.data import Dataset
import json
import gzip
from utils.features import encode_chunk

class PokerDataset(Dataset):
    def __init__(self, file_path):
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

    def __len__(self):
        return len(self.labeled_chunks)

    def __getitem__(self, idx):
        chunk_entry = self.labeled_chunks[idx]
        chunks = chunk_entry.get("hands", chunk_entry)

        sequence = []
        for chunk in chunks:
            sequence.append(encode_chunk(chunk))

        x = torch.tensor(sequence, dtype=torch.float32)

        label = 1 if chunk_entry.get("is_bot", False) else 0
        y = torch.tensor([label], dtype=torch.float32)

        return x, y
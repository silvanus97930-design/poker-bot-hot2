import torch
from torch.utils.data import Dataset
import json
from utils.features import encode_chunk

class PokerDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data["chunks"])

    def __getitem__(self, idx):
        chunks = self.data["chunks"][idx]

        sequence = []
        for chunk in chunks:
            sequence.append(encode_chunk(chunk))

        x = torch.tensor(sequence, dtype=torch.float32)

        label = 1 if chunks[0]["label"] == "bot" else 0
        y = torch.tensor([label], dtype=torch.float32)

        return x, y
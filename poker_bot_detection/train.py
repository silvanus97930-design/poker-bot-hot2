import torch
from torch.utils.data import DataLoader
from models.gru_model import GRUClassifier, LabelSmoothingBCE
from utils.dataset import PokerDataset, poker_collate_fn
import config

def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = PokerDataset("data/train.json")
    val_dataset = PokerDataset("data/val.json")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=poker_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        collate_fn=poker_collate_fn,
    )

    model = GRUClassifier(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY
    )

    criterion = LabelSmoothingBCE(0.1)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.EPOCHS):

        model.train()
        for x, lengths, y in train_loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x, lengths=lengths)
            loss = criterion(logits, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for x, lengths, y in val_loader:
                x, lengths, y = x.to(device), lengths.to(device), y.to(device)
                logits = model(x, lengths=lengths)
                val_loss += criterion(logits, y).item()

        print(f"Epoch {epoch}, Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            patience_counter += 1

        if patience_counter >= config.PATIENCE:
            print("Early stopping")
            break


if __name__ == "__main__":
    train()
import torch
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from models.gru_model import GRUClassifier, LabelSmoothingBCE
from utils.dataset import (
    PokerDataset,
    poker_collate_fn,
    compute_balance_stats,
    train_sample_weights,
    fit_feature_normalization,
    save_feature_norm,
)
import config


def _print_balance_report(split_name: str, stats: dict) -> None:
    h = stats["hands_per_chunk"]
    print(
        f"[balance] {split_name}: chunks={stats['num_chunks']} "
        f"bot={stats['bot_count']} human={stats['human_count']} "
        f"minority_ratio={stats['minority_ratio']:.3f} "
        f"hands/chunk avg={h['avg']:.1f} min={h['min']} max={h['max']}"
    )


def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    norm_path = (
        Path(config.FEATURE_NORM_PATH)
        if getattr(config, "FEATURE_NORM_PATH", "")
        else Path(__file__).resolve().parent / "feature_norm.pt"
    )

    train_for_norm = PokerDataset(config.DATA_PATH, split="train")
    feature_mean, feature_std = fit_feature_normalization(
        train_for_norm,
        config.INPUT_DIM,
        eps=config.FEATURE_NORM_EPS,
    )
    save_feature_norm(norm_path, feature_mean, feature_std, config.INPUT_DIM)
    print(
        f"[norm] fit on train only: saved mean/std shape={tuple(feature_mean.shape)} "
        f"to {norm_path}"
    )

    train_dataset = PokerDataset(
        config.DATA_PATH,
        split="train",
        feature_mean=feature_mean,
        feature_std=feature_std,
        norm_eps=config.FEATURE_NORM_EPS,
    )
    val_dataset = PokerDataset(
        config.DATA_PATH,
        split="validation",
        feature_mean=feature_mean,
        feature_std=feature_std,
        norm_eps=config.FEATURE_NORM_EPS,
    )

    train_stats = compute_balance_stats(train_dataset)
    val_stats = compute_balance_stats(val_dataset)
    _print_balance_report("train", train_stats)
    _print_balance_report("val", val_stats)

    imbalance_train = train_stats["minority_ratio"] < config.BALANCE_IMBALANCE_RATIO_THRESHOLD
    strategy = config.BALANCE_STRATEGY
    if strategy == "auto":
        strategy = "pos_weight" if imbalance_train else "none"

    pos_weight_tensor = None
    train_sampler = None
    shuffle_train = True

    if imbalance_train and strategy == "pos_weight":
        n_bot = train_stats["bot_count"]
        n_human = train_stats["human_count"]
        if n_bot > 0 and n_human > 0:
            # Up-weight positive class (bot) when bots are rare; symmetric when humans are rare.
            pos_weight_tensor = torch.tensor([n_human / n_bot], dtype=torch.float32)
            print(
                f"[balance] imbalanced (minority_ratio={train_stats['minority_ratio']:.3f} "
                f"< {config.BALANCE_IMBALANCE_RATIO_THRESHOLD}): using pos_weight={pos_weight_tensor.item():.4f}"
            )
        else:
            print("[balance] imbalanced but one class is empty; cannot set pos_weight.")
    elif imbalance_train and strategy == "weighted_sampler":
        train_sampler = WeightedRandomSampler(
            train_sample_weights(train_dataset),
            num_samples=len(train_dataset),
            replacement=True,
        )
        shuffle_train = False
        print(
            f"[balance] imbalanced: using WeightedRandomSampler "
            f"(minority_ratio={train_stats['minority_ratio']:.3f})"
        )
    elif imbalance_train:
        print("[balance] imbalanced but BALANCE_STRATEGY=none; no reweighting applied.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle_train,
        sampler=train_sampler,
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

    criterion = LabelSmoothingBCE(0.1, pos_weight=pos_weight_tensor)

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
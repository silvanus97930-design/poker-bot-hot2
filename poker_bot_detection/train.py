import torch
from pathlib import Path
import gzip
import json
from torch.utils.data import DataLoader, WeightedRandomSampler
from models.gru_model import GRUClassifier, LabelSmoothingBCE
from utils.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from utils.dataset import (
    PokerDataset,
    poker_collate_fn,
    compute_balance_stats,
    train_sample_weights,
    fit_feature_normalization,
    save_feature_norm,
)
import config


def _load_dataset_hash(data_path: str) -> str:
    path = Path(data_path)
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    return str(payload.get("dataset_hash", ""))


def _save_preprocessing_artifact(
    artifact_path: Path,
    *,
    data_path: str,
    norm_path: Path,
    input_dim: int,
) -> None:
    dataset_hash = _load_dataset_hash(data_path)
    artifact = {
        "artifact_version": 1,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "input_dim": int(input_dim),
        "normalization": {
            "type": "standardization",
            "fit_split": "train",
            "norm_file": str(norm_path),
            "eps": float(config.FEATURE_NORM_EPS),
        },
        "dataset": {
            "data_path": str(data_path),
            "dataset_hash": dataset_hash,
        },
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=True)
    print(f"[artifact] saved preprocessing artifact to {artifact_path}")


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
    artifact_path = (
        Path(config.PREPROCESSING_ARTIFACT_PATH)
        if getattr(config, "PREPROCESSING_ARTIFACT_PATH", "")
        else Path(__file__).resolve().parent / "preprocessing_artifact.json"
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
    if len(FEATURE_NAMES) != config.INPUT_DIM:
        raise ValueError(
            f"FEATURE_NAMES has {len(FEATURE_NAMES)} entries but INPUT_DIM={config.INPUT_DIM}."
        )
    _save_preprocessing_artifact(
        artifact_path,
        data_path=config.DATA_PATH,
        norm_path=norm_path,
        input_dim=config.INPUT_DIM,
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
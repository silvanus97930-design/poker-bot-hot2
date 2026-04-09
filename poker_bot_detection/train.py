import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from pathlib import Path
import gzip
import json
from torch.utils.data import DataLoader, WeightedRandomSampler
from models.gru_model import GRUTransformerClassifier, LabelSmoothingBCE
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

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


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


def _assert_dataset_split(dataset: PokerDataset, expected_split: str, name: str) -> None:
    for i, entry in enumerate(dataset.labeled_chunks):
        if not isinstance(entry, dict):
            continue
        split = entry.get("split")
        if split is not None and split != expected_split:
            raise ValueError(
                f"{name} contains non-{expected_split} entry at index {i}: split={split}"
            )


def _run_pretrain_sanity_checks(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_dataset: PokerDataset,
    val_dataset: PokerDataset,
    device: torch.device,
) -> None:
    _assert_dataset_split(train_dataset, "train", "train_dataset")
    _assert_dataset_split(val_dataset, "validation", "val_dataset")
    print("[sanity] split check passed (validation loader reads validation split only).")

    sample_batch = next(iter(train_loader))
    x, lengths, y = sample_batch
    if x.dim() != 3:
        raise ValueError(f"Expected x rank 3 [batch, seq, dim], got shape={tuple(x.shape)}")
    if x.shape[-1] != config.INPUT_DIM:
        raise ValueError(
            f"Batch input_dim mismatch: x.shape[-1]={x.shape[-1]} vs config.INPUT_DIM={config.INPUT_DIM}"
        )
    if y.dim() != 2 or y.shape[-1] != 1:
        raise ValueError(f"Expected y shape [batch, 1], got {tuple(y.shape)}")

    model.eval()
    with torch.no_grad():
        logits = model(x.to(device), lengths=lengths.to(device))
    if logits.shape != y.shape:
        raise ValueError(
            f"Logits/label shape mismatch: logits={tuple(logits.shape)} vs y={tuple(y.shape)}"
        )
    print(
        f"[sanity] one-batch forward passed | x={tuple(x.shape)} lengths={tuple(lengths.shape)} "
        f"logits={tuple(logits.shape)} labels={tuple(y.shape)}"
    )

    val_batch = next(iter(val_loader))
    vx, vl, vy = val_batch
    if vx.shape[-1] != config.INPUT_DIM:
        raise ValueError(
            f"Validation input_dim mismatch: vx.shape[-1]={vx.shape[-1]} vs config.INPUT_DIM={config.INPUT_DIM}"
        )
    print(
        f"[sanity] validation batch shape check passed | x={tuple(vx.shape)} "
        f"lengths={tuple(vl.shape)} labels={tuple(vy.shape)}"
    )


def _resolve_data_path() -> str:
    return str(getattr(config, "DATA_PATH_OVERRIDE", "") or config.DATA_PATH)


def _progress_iter(loader, *, desc: str, leave: bool):
    if tqdm is None:
        return loader
    return tqdm(loader, desc=desc, leave=leave, dynamic_ncols=True)


def train():

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training run, but no GPU was detected.")
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    if getattr(config, "REQUIRE_RTX_3090", True) and "3090" not in gpu_name:
        raise RuntimeError(
            f"Expected RTX 3090 but detected '{gpu_name}'. "
            "Set REQUIRE_RTX_3090=False in config to bypass."
        )
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print(f"[device] using GPU: {gpu_name}")
    print(
        f"[device] tf32 matmul={torch.backends.cuda.matmul.allow_tf32} "
        f"cudnn_tf32={torch.backends.cudnn.allow_tf32}"
    )

    data_path = _resolve_data_path()

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

    train_for_norm = PokerDataset(data_path, split="train")
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
        data_path=data_path,
        norm_path=norm_path,
        input_dim=config.INPUT_DIM,
    )

    train_dataset = PokerDataset(
        data_path,
        split="train",
        feature_mean=feature_mean,
        feature_std=feature_std,
        norm_eps=config.FEATURE_NORM_EPS,
    )
    val_dataset = PokerDataset(
        data_path,
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

    num_workers = int(getattr(config, "NUM_WORKERS", 8))
    pin_memory = bool(getattr(config, "PIN_MEMORY", True))
    persistent_workers = bool(getattr(config, "PERSISTENT_WORKERS", True)) and num_workers > 0
    prefetch_factor = int(getattr(config, "PREFETCH_FACTOR", 2))

    train_loader_kwargs = {
        "batch_size": config.BATCH_SIZE,
        "shuffle": shuffle_train,
        "sampler": train_sampler,
        "collate_fn": poker_collate_fn,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    val_loader_kwargs = {
        "batch_size": config.BATCH_SIZE,
        "collate_fn": poker_collate_fn,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = prefetch_factor
        val_loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        **train_loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        **val_loader_kwargs,
    )

    model = GRUTransformerClassifier(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        gru_layers=config.NUM_LAYERS,
        tf_layers=getattr(config, "TF_LAYERS", 2),
        nhead=getattr(config, "NHEAD", 4),
        ff_mult=getattr(config, "FF_MULT", 4),
        dropout=config.DROPOUT,
        max_len=getattr(config, "MAX_SEQ_LEN", 2048),
        bidirectional_gru=getattr(config, "BIDIRECTIONAL_GRU", True),
        use_attention_pool=getattr(config, "USE_ATTENTION_POOL", True),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(2, int(config.PATIENCE // 3)),
        min_lr=1e-6,
    )

    criterion = LabelSmoothingBCE(
        getattr(config, "LABEL_SMOOTHING", 0.05),
        pos_weight=pos_weight_tensor,
    )
    amp_enabled = bool(getattr(config, "AMP_ENABLED", True))
    scaler = GradScaler("cuda", enabled=amp_enabled)
    _run_pretrain_sanity_checks(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
    )

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.EPOCHS):

        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        train_iter = _progress_iter(
            train_loader,
            desc=f"Epoch {epoch + 1}/{config.EPOCHS} [train]",
            leave=False,
        )
        for x, lengths, y in train_iter:
            x = x.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=amp_enabled):
                logits = model(x, lengths=lengths)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(getattr(config, "GRAD_CLIP_NORM", 1.0)),
            )
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += float(loss.item())
            train_batches += 1
            if tqdm is not None:
                train_iter.set_postfix(loss=f"{loss.item():.4f}")

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0

        val_iter = _progress_iter(
            val_loader,
            desc=f"Epoch {epoch + 1}/{config.EPOCHS} [val]",
            leave=False,
        )
        with torch.no_grad():
            for x, lengths, y in val_iter:
                x = x.to(device, non_blocking=True)
                lengths = lengths.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with autocast("cuda", enabled=amp_enabled):
                    logits = model(x, lengths=lengths)
                    loss = criterion(logits, y)
                val_loss_sum += float(loss.item())
                val_batches += 1
                if tqdm is not None:
                    val_iter.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = train_loss_sum / max(1, train_batches)
        val_loss = val_loss_sum / max(1, val_batches)
        print(
            f"Epoch {epoch + 1}/{config.EPOCHS} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"best_val={best_val_loss:.4f} lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        scheduler.step(val_loss)

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
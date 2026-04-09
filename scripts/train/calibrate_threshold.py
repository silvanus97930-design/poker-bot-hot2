#!/usr/bin/env python3
"""Calibrate a robust decision threshold across public benchmark files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "poker_bot_detection") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "poker_bot_detection"))

import poker_bot_detection.config as config
from poker_bot_detection.models.gru_model import GRUTransformerClassifier
from poker_bot_detection.utils.dataset import PokerDataset, load_feature_norm, poker_collate_fn


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=REPO_ROOT / "best_model.pt")
    parser.add_argument(
        "--norm-path", type=Path, default=REPO_ROOT / "poker_bot_detection" / "feature_norm.pt"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            str(REPO_ROOT / "data" / "public_miner_benchmark_seed101.json.gz"),
            str(REPO_ROOT / "data" / "public_miner_benchmark_seed202.json.gz"),
            str(REPO_ROOT / "data" / "public_miner_benchmark_merged_seed44.json.gz"),
        ],
        help="Benchmark files to evaluate (validation split).",
    )
    parser.add_argument("--threshold-min", type=float, default=0.01)
    parser.add_argument("--threshold-max", type=float, default=0.99)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--min-precision", type=float, default=0.98)
    parser.add_argument("--output-json", type=Path, default=REPO_ROOT / "runs" / "threshold_calibration.json")
    return parser.parse_args()


def _build_model(device: torch.device) -> GRUTransformerClassifier:
    return GRUTransformerClassifier(
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


def _predict_scores(
    *,
    data_path: Path,
    model: GRUTransformerClassifier,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    ds = PokerDataset(
        str(data_path),
        split="validation",
        feature_mean=mean,
        feature_std=std,
        norm_eps=config.FEATURE_NORM_EPS,
    )
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, collate_fn=poker_collate_fn)
    ys: List[float] = []
    ps: List[float] = []
    with torch.no_grad():
        for x, lengths, y in loader:
            x = x.to(device)
            lengths = lengths.to(device)
            logits = model(x, lengths=lengths)
            prob = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
            label = y.squeeze(-1).cpu().numpy()
            ys.extend(label.tolist())
            ps.extend(prob.tolist())
    return {"y_true": np.array(ys).astype(int), "y_score": np.array(ps)}


def _metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (y_score >= threshold).astype(int)
    return {
        "threshold": threshold,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
    }


def main() -> None:
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std, _ = load_feature_norm(args.norm_path)

    model = _build_model(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    per_dataset = []
    combined_y = []
    combined_p = []
    for ds_path_raw in args.datasets:
        ds_path = Path(ds_path_raw)
        if not ds_path.exists():
            print(f"[skip] missing dataset: {ds_path}")
            continue
        out = _predict_scores(data_path=ds_path, model=model, mean=mean, std=std, device=device)
        y_true = out["y_true"]
        y_score = out["y_score"]
        combined_y.append(y_true)
        combined_p.append(y_score)
        per_dataset.append(
            {
                "dataset": str(ds_path),
                "n_validation_chunks": int(y_true.size),
                "pr_auc": float(average_precision_score(y_true, y_score)),
                "roc_auc": float(roc_auc_score(y_true, y_score)),
            }
        )

    if not per_dataset:
        raise RuntimeError("No valid datasets were evaluated.")

    y_true = np.concatenate(combined_y)
    y_score = np.concatenate(combined_p)
    thresholds = np.arange(args.threshold_min, args.threshold_max + 1e-9, args.threshold_step)
    rows = [_metrics_at_threshold(y_true, y_score, float(thr)) for thr in thresholds]

    # Select robust threshold:
    # 1) max F1 among thresholds that satisfy precision >= min_precision
    # 2) if none satisfy, max F1 overall
    eligible = [r for r in rows if r["precision"] >= args.min_precision]
    chosen = max(eligible, key=lambda r: r["f1"]) if eligible else max(rows, key=lambda r: r["f1"])

    result = {
        "model_path": str(args.model_path),
        "norm_path": str(args.norm_path),
        "datasets": per_dataset,
        "combined": {
            "n_validation_chunks": int(y_true.size),
            "pr_auc": float(average_precision_score(y_true, y_score)),
            "roc_auc": float(roc_auc_score(y_true, y_score)),
        },
        "selection_policy": {
            "min_precision": float(args.min_precision),
            "criterion": "max_f1_subject_to_precision",
        },
        "recommended_threshold": float(chosen["threshold"]),
        "recommended_metrics": {
            "f1": float(chosen["f1"]),
            "precision": float(chosen["precision"]),
            "recall": float(chosen["recall"]),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"combined_pr_auc={result['combined']['pr_auc']:.6f}")
    print(f"combined_roc_auc={result['combined']['roc_auc']:.6f}")
    print(f"recommended_threshold={result['recommended_threshold']:.4f}")
    print(
        f"recommended_f1={result['recommended_metrics']['f1']:.6f} "
        f"precision={result['recommended_metrics']['precision']:.6f} "
        f"recall={result['recommended_metrics']['recall']:.6f}"
    )
    print(f"saved={args.output_json}")


if __name__ == "__main__":
    main()


#!/usr/bin/env bash
# Multi-seed public benchmark build → train → eval, saving a distinct checkpoint per seed.
#
# Usage (from repo root):
#   chmod +x scripts/train/multi_seed_train_eval.sh
#   ./scripts/train/multi_seed_train_eval.sh
#
# Optional env overrides:
#   SEEDS="44 101 202" ./scripts/train/multi_seed_train_eval.sh
#   CHUNK_COUNT=300 VAL_RATIO=0.25 ./scripts/train/multi_seed_train_eval.sh
#
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

# shellcheck source=/dev/null
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

mkdir -p runs
RESULTS_CSV="runs/multi_seed_results.csv"
echo "seed,benchmark_path,model_path,norm_path,auc,precision,recall,f1,tn,fp,fn,tp" > "$RESULTS_CSV"

SEEDS="${SEEDS:-44 101 202}"
CHUNK_COUNT="${CHUNK_COUNT:-300}"
VAL_RATIO="${VAL_RATIO:-0.25}"

for SEED in $SEEDS; do
  echo "=== Seed: $SEED ==="

  BENCH="data/public_miner_benchmark_seed${SEED}.json.gz"
  MODEL_OUT="runs/best_model_seed${SEED}.pt"
  NORM_OUT="runs/feature_norm_seed${SEED}.pt"
  ARTIFACT_OUT="runs/preprocessing_artifact_seed${SEED}.json"

  if ! python scripts/publish/publish_public_benchmark.py \
    --skip-wandb \
    --seed "$SEED" \
    --chunk-count "$CHUNK_COUNT" \
    --validation-ratio "$VAL_RATIO" \
    --output-path "$BENCH"
  then
    echo "benchmark failed for $SEED"
    continue
  fi

  if ! (
    cd poker_bot_detection || exit 1
    PYTHONPATH="$REPO_ROOT" DATA_PATH="../$BENCH" python - <<'PY'
import os
import poker_bot_detection.config as config
config.DATA_PATH = os.environ["DATA_PATH"]
from poker_bot_detection.train import train
train()
PY
  ); then
    echo "train failed for $SEED"
    continue
  fi

  cp -f poker_bot_detection/best_model.pt "$MODEL_OUT"
  cp -f poker_bot_detection/feature_norm.pt "$NORM_OUT"
  cp -f poker_bot_detection/preprocessing_artifact.json "$ARTIFACT_OUT"
  echo "[runs] saved $MODEL_OUT , $NORM_OUT , $ARTIFACT_OUT"

  if ! PYTHONPATH="$REPO_ROOT" \
    DATA_PATH="$BENCH" \
    SEED="$SEED" \
    MODEL_PATH="$MODEL_OUT" \
    NORM_PATH="$NORM_OUT" \
    python - <<'PY' >> "$RESULTS_CSV"
import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix

import poker_bot_detection.config as config
from poker_bot_detection.models.gru_model import GRUClassifier
from poker_bot_detection.utils.dataset import PokerDataset, poker_collate_fn, load_feature_norm

seed = os.environ["SEED"]
bench = os.environ["DATA_PATH"]
model_path = os.environ["MODEL_PATH"]
norm_path = os.environ["NORM_PATH"]
config.DATA_PATH = bench

mean, std, saved_dim = load_feature_norm(norm_path)
if saved_dim is not None and int(saved_dim) != int(config.INPUT_DIM):
    raise ValueError(f"input_dim mismatch: norm={saved_dim}, config={config.INPUT_DIM}")

val_ds = PokerDataset(
    config.DATA_PATH,
    split="validation",
    feature_mean=mean,
    feature_std=std,
    norm_eps=config.FEATURE_NORM_EPS,
)
val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, collate_fn=poker_collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GRUClassifier(
    input_dim=config.INPUT_DIM,
    hidden_dim=config.HIDDEN_DIM,
    num_layers=config.NUM_LAYERS,
    dropout=config.DROPOUT,
).to(device)
state = torch.load(model_path, map_location=device)
model.load_state_dict(state)
model.eval()

y_true, y_prob = [], []
with torch.no_grad():
    for x, lengths, y in val_loader:
        x, lengths = x.to(device), lengths.to(device)
        probs = torch.sigmoid(model(x, lengths=lengths)).squeeze(1).cpu().tolist()
        y_prob.extend(probs)
        y_true.extend(y.squeeze(1).cpu().int().tolist())

auc = roc_auc_score(y_true, y_prob)
y_pred = [1 if p >= 0.5 else 0 for p in y_prob]
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="binary", zero_division=0
)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

print(
    f"{seed},{bench},{model_path},{norm_path},"
    f"{auc:.6f},{precision:.6f},{recall:.6f},{f1:.6f},{tn},{fp},{fn},{tp}"
)
PY
  then
    echo "eval failed for $SEED"
    continue
  fi

done

echo "Saved: $RESULTS_CSV"
cat "$RESULTS_CSV"

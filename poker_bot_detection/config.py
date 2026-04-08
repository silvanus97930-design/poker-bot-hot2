INPUT_DIM = 31  # must match encode_hand() output size in utils/features.py
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3

LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 7
SEQ_LEN = 40

# Public benchmark: one file with labeled_chunks[].split == "train" | "validation"

# DATA_PATH = "data/public_miner_benchmark.json.gz"
DATA_PATH = "data/public_miner_benchmark_seed101.json.gz"


# If min(bot_count, human_count) / max(bot_count, human_count) < this, treat as imbalanced.
BALANCE_IMBALANCE_RATIO_THRESHOLD = 0.35
# "pos_weight" | "weighted_sampler" | "none" — "auto" picks pos_weight when imbalanced.
BALANCE_STRATEGY = "auto"

# Per-feature standardization (mean/std fit on train split only; same tensors for val).
FEATURE_NORM_EPS = 1e-8
# If empty, train.py writes next to train.py as feature_norm.pt
FEATURE_NORM_PATH = ""

# Reproducibility artifact with schema/version, dataset hash, and norm metadata.
# If empty, train.py writes next to train.py as preprocessing_artifact.json
PREPROCESSING_ARTIFACT_PATH = ""
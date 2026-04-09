INPUT_DIM = 31  # must match encode_hand() output size in utils/features.py
HIDDEN_DIM = 192
NUM_LAYERS = 2
DROPOUT = 0.15

LR = 2e-4
WEIGHT_DECAY = 3e-4
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 12
SEQ_LEN = 40

# Public benchmark: one file with labeled_chunks[].split == "train" | "validation"

# DATA_PATH = "data/public_miner_benchmark.json.gz"
DATA_PATH = "data/external/public_miner_benchmark_merged_seed44.json.gz"
DATA_PATH_OVERRIDE = "data/public_miner_benchmark_strong_seed44.json.gz"

# Enforce that training uses RTX 3090 unless explicitly disabled.
REQUIRE_RTX_3090 = True

# GRU+Transformer model settings
TF_LAYERS = 2
NHEAD = 6
FF_MULT = 4
MAX_SEQ_LEN = 2048
BIDIRECTIONAL_GRU = True
USE_ATTENTION_POOL = True
INFERENCE_THRESHOLD = 0.02

# Throughput settings for RTX 3090
AMP_ENABLED = True
NUM_WORKERS = 8
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2
LABEL_SMOOTHING = 0.03
GRAD_CLIP_NORM = 1.0


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
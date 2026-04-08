import torch
from pathlib import Path

from models.gru_model import GRUClassifier
from utils.dataset import apply_feature_normalization, load_feature_norm
import config

def load_model():
    model = GRUClassifier(config.INPUT_DIM)
    model.load_state_dict(torch.load("best_model.pt"))
    model.eval()
    return model


def load_feature_norm_if_present():
    path = Path(__file__).resolve().parent / "feature_norm.pt"
    if not path.exists():
        return None, None
    mean, std, _ = load_feature_norm(path)
    return mean, std


def predict(model, x, lengths=None, feature_mean=None, feature_std=None, norm_eps=None):
    """
    x: [batch, seq, dim] raw hand features unless mean/std provided (then same as training).
    lengths: optional sequence lengths for GRU packing (matches training).
    """
    eps = norm_eps if norm_eps is not None else getattr(config, "FEATURE_NORM_EPS", 1e-8)
    if feature_mean is not None and feature_std is not None:
        x = apply_feature_normalization(x, feature_mean, feature_std, eps)
    with torch.no_grad():
        logits = model(x, lengths=lengths)
        probs = torch.sigmoid(logits)
        return probs, probs > 0.5
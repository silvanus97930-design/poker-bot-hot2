import torch
from models.gru_model import GRUClassifier
import config

def load_model():
    model = GRUClassifier(config.INPUT_DIM)
    model.load_state_dict(torch.load("best_model.pt"))
    model.eval()
    return model


def predict(model, x):
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)
        return probs, probs > 0.5
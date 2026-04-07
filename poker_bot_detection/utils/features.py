import numpy as np

def encode_chunk(chunk):
    features = []

    # Example features (expand this!)
    features.append(chunk["metadata"]["sb"])
    features.append(chunk["metadata"]["bb"])
    features.append(len(chunk["players"]))

    # Pot size
    features.append(chunk["outcome"]["total_pot"])

    # Number of actions
    features.append(len(chunk["actions"]))

    return np.array(features, dtype=np.float32)
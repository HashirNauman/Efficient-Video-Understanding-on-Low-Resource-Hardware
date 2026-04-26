import os
import pickle
import warnings

import numpy as np
import torch
import torchvision.models as models


CACHE_DIR = "feature_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def save_cached_features(key, features):
    with open(f"{CACHE_DIR}/{key}.pkl", "wb") as f:
        pickle.dump(features, f)


def load_cached_features(key):
    path = f"{CACHE_DIR}/{key}.pkl"
    if os.path.exists(path):
        print(f"Loading cached features: {key}")
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def extract_cnn_features(frames, batch_size=32, device=None):
    print("CNN feature extraction (batched)...")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = models.mobilenet_v2(weights="DEFAULT")
    except Exception as exc:
        warnings.warn(f"Falling back to randomly initialized MobileNetV2: {exc}")
        model = models.mobilenet_v2(weights=None)

    model = model.features.to(device)
    model.eval()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    features = []

    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]

            try:
                x = torch.tensor(batch).permute(0, 3, 1, 2).float().to(device)
                x = (x - mean) / std
                f = model(x)
            except RuntimeError as exc:
                if device.type == "cuda":
                    warnings.warn(f"CUDA feature extraction failed, retrying on CPU: {exc}")
                    return extract_cnn_features(
                        frames,
                        batch_size=batch_size,
                        device=torch.device("cpu"),
                    )
                raise

            f = torch.mean(f, dim=[2, 3])
            features.append(f.cpu().numpy())

    return np.vstack(features).astype(np.float32)


def combine_features(cnn_features, skeleton_features):
    min_len = min(len(cnn_features), len(skeleton_features))
    return np.concatenate(
        [cnn_features[:min_len], skeleton_features[:min_len]],
        axis=1
    )

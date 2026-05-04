"""
features.py
===========
CNN feature extraction, temporal augmentation, sequence building, and
feature caching.

Design decisions:
- MobileNetV2 pretrained on ImageNet is used as a frozen feature extractor.
  Its final spatial feature map is globally average-pooled to a 1280-d vector
  per frame. No fine-tuning is performed (out of scope for this benchmark).
- Temporal deltas (frame[t] - frame[t-1]) are appended to each frame feature,
  doubling the dimensionality to 2560. This encodes motion information without
  requiring explicit optical flow computation.
- `build_video_sequences` pads short videos (< max_frames_per_video) with
  zero vectors so that np.stack produces a consistent (N_videos, T, D) tensor.
  Padding masks are returned so sequence models can ignore padded timesteps.
- Cache keys encode dataset, split, pipeline version, and all frame-selection
  hyperparameters, ensuring stale caches from prior experiments are never
  silently reused.
"""

import os
import pickle
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torchvision.models as models

from preprocessing import MAX_FRAMES_PER_VIDEO


CACHE_DIR = "feature_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def save_cached_features(key: str, features) -> None:
    path = os.path.join(CACHE_DIR, f"{key}.pkl")
    with open(path, "wb") as f:
        pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_cached_features(key: str):
    path = os.path.join(CACHE_DIR, f"{key}.pkl")
    if os.path.exists(path):
        print(f"Loading cached features: {key}")
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


# ---------------------------------------------------------------------------
# CNN feature extraction
# ---------------------------------------------------------------------------

def extract_cnn_features(
    frames: np.ndarray,
    batch_size: int = 32,
    device=None,
) -> np.ndarray:
    """Extract MobileNetV2 features for each frame.

    Parameters
    ----------
    frames     : float32 array (N, H, W, 3), values in [0, 1]
    batch_size : GPU mini-batch size; reduced automatically on OOM
    device     : torch.device; auto-detected if None

    Returns
    -------
    features : float32 array (N, 1280)
    """
    print("CNN feature extraction (MobileNetV2, pretrained ImageNet)...")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = models.mobilenet_v2(weights="DEFAULT")
    except Exception as exc:
        warnings.warn(
            f"Could not load pretrained MobileNetV2 weights: {exc}. "
            "Falling back to random initialisation — results will not be "
            "comparable to standard benchmarks."
        )
        model = models.mobilenet_v2(weights=None)

    # Use only the convolutional backbone (output: 1280 channels, 4×4 spatial)
    model = model.features.to(device)
    model.eval()

    # ImageNet normalisation constants
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    all_features = []

    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch = frames[i : i + batch_size]

            try:
                x = torch.tensor(batch).permute(0, 3, 1, 2).float().to(device)
                x = (x - mean) / std
                f = model(x)
            except RuntimeError as exc:
                if device.type == "cuda":
                    warnings.warn(
                        f"CUDA OOM during CNN extraction, retrying on CPU: {exc}"
                    )
                    return extract_cnn_features(
                        frames, batch_size=batch_size, device=torch.device("cpu")
                    )
                raise

            # Global average pool over spatial dims → (B, 1280)
            f = torch.mean(f, dim=[2, 3])
            all_features.append(f.cpu().numpy())

    return np.vstack(all_features).astype(np.float32)


# ---------------------------------------------------------------------------
# Temporal augmentation
# ---------------------------------------------------------------------------

def augment_with_temporal_deltas(
    features: np.ndarray,
    video_ids: np.ndarray,
) -> np.ndarray:
    """Append frame-to-frame deltas to every feature vector.

    For frame t:  output[t] = [features[t] | features[t] - features[t-1]]
    First frame:  delta = zeros (no prior frame).

    This doubles the feature dimensionality and provides explicit motion
    information without requiring optical flow.

    Parameters
    ----------
    features  : float32 (N, D)
    video_ids : int    (N,)  — used to avoid computing deltas across videos

    Returns
    -------
    augmented : float32 (N, 2D)
    """
    D = features.shape[1]
    augmented = np.zeros((len(features), D * 2), dtype=np.float32)

    for vid in np.unique(video_ids):
        idxs = np.where(video_ids == vid)[0]
        vid_features = features[idxs]
        delta = np.zeros_like(vid_features)
        if len(vid_features) > 1:
            delta[1:] = vid_features[1:] - vid_features[:-1]
        augmented[idxs] = np.concatenate([vid_features, delta], axis=1)

    return augmented


# ---------------------------------------------------------------------------
# Sequence building
# ---------------------------------------------------------------------------

def build_video_sequences(
    features: np.ndarray,
    video_ids: np.ndarray,
    frame_labels: np.ndarray,
    max_len: int = MAX_FRAMES_PER_VIDEO,
) -> tuple:
    """Stack per-video frame features into a fixed-length sequence tensor.

    Videos with fewer than `max_len` frames are zero-padded at the end.
    A boolean padding mask is returned: True = real frame, False = pad.

    Parameters
    ----------
    features     : float32 (N, D)
    video_ids    : int    (N,)
    frame_labels : int    (N,)
    max_len      : sequence length (matches MAX_FRAMES_PER_VIDEO)

    Returns
    -------
    sequences : float32 (V, max_len, D)
    labels    : int    (V,)
    masks     : bool   (V, max_len)  — True for real, False for pad
    """
    D = features.shape[1]
    sequences, labels, masks = [], [], []

    for vid in sorted(np.unique(video_ids)):
        idxs = np.where(video_ids == vid)[0]
        video_labels = np.unique(frame_labels[idxs])

        if len(video_labels) != 1:
            raise ValueError(
                f"Inconsistent frame labels for video id {vid}: {video_labels}. "
                "Each video must have a single class label."
            )

        vid_feat = features[idxs]
        T = len(vid_feat)
        seq = np.zeros((max_len, D), dtype=np.float32)
        mask = np.zeros(max_len, dtype=bool)

        if T >= max_len:
            seq[:] = vid_feat[:max_len]
            mask[:] = True
        else:
            seq[:T] = vid_feat
            mask[:T] = True

        sequences.append(seq)
        labels.append(int(video_labels[0]))
        masks.append(mask)

    return (
        np.stack(sequences).astype(np.float32),
        np.array(labels, dtype=np.int64),
        np.stack(masks),
    )


# ---------------------------------------------------------------------------
# Multimodal feature fusion
# ---------------------------------------------------------------------------

def combine_features(
    cnn_features: np.ndarray,
    skeleton_features: np.ndarray,
) -> np.ndarray:
    """Concatenate CNN and skeleton features along the feature axis.

    Trims to the shorter array if lengths differ (should not happen in normal
    use, but guards against frame-skip mismatches).
    """
    min_len = min(len(cnn_features), len(skeleton_features))
    return np.concatenate(
        [cnn_features[:min_len], skeleton_features[:min_len]],
        axis=1,
    )

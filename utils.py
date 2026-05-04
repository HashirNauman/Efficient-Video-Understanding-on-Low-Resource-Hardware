"""
utils.py
========
Evaluation, latency measurement, artefact saving, and results logging.

Latency reporting
-----------------
All models are measured identically: one sample at a time, same number of
warm-up calls, CUDA synchronisation before and after the timed window.
For sequence models the 'sample' is a single video sequence (shape 1 × T × D);
for frame models it is a single feature vector (shape 1 × D). This makes
efficiency_score (accuracy / latency_ms) directly comparable across modes
because all modes measure the cost of classifying one video.
"""

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
)


# ---------------------------------------------------------------------------
# GPU memory helpers
# ---------------------------------------------------------------------------

def get_peak_gpu_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024 ** 2))


def reset_gpu_memory_tracking() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_predictions(y_true, y_pred, split_name: str = "TEST") -> float:
    accuracy = float(accuracy_score(y_true, y_pred))
    print(f"{split_name} Accuracy: {accuracy:.4f}")
    return accuracy


# ---------------------------------------------------------------------------
# Latency / throughput
# ---------------------------------------------------------------------------

def measure_latency(model, X, num_samples: int = 100, masks=None) -> tuple:
    """Measure per-sample inference latency in milliseconds.

    Parameters
    ----------
    model      : object with a predict() method
    X          : array of shape (N, ...) — one sample = one video
    num_samples: how many samples to time (uses first min(N, num_samples))
    masks      : optional padding masks for sequence models (N, T)

    Returns
    -------
    latency_ms : float — mean latency per sample (ms)
    fps        : float — samples per second
    """
    print("Measuring latency...")

    n = min(num_samples, len(X))
    if n == 0:
        return 0.0, 0.0

    X_sample = X[:n]
    masks_sample = masks[:n] if masks is not None else None

    # CUDA warm-up (5 samples, not timed)
    warmup = min(5, n)
    for i in range(warmup):
        xi = X_sample[i : i + 1]
        mi = masks_sample[i : i + 1] if masks_sample is not None else None
        if mi is not None:
            model.predict(xi, mi)
        else:
            model.predict(xi)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()

    for i in range(n):
        xi = X_sample[i : i + 1]
        mi = masks_sample[i : i + 1] if masks_sample is not None else None
        if mi is not None:
            model.predict(xi, mi)
        else:
            model.predict(xi)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    if elapsed <= 0:
        return 0.0, 0.0

    latency_ms = (elapsed / n) * 1000.0
    fps = n / elapsed

    print(f"Latency: {latency_ms:.3f} ms | FPS: {fps:.2f}")
    return float(latency_ms), float(fps)


# ---------------------------------------------------------------------------
# Artefact: training curves
# ---------------------------------------------------------------------------

def save_training_curve(history: dict, mode: str, dataset: str):
    """Save a loss + accuracy plot for models that record training history."""
    if not history or not history.get("train_loss"):
        return None

    os.makedirs("artifacts/training_curves", exist_ok=True)
    path = os.path.join(
        "artifacts", "training_curves", f"{dataset}_{mode}_curve.png"
    )

    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="train")
    if history.get("val_loss"):
        ax.plot(
            epochs[:len(history["val_loss"])],
            history["val_loss"],
            label="val",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{dataset} | {mode} — Loss")
    ax.legend()

    ax = axes[1]
    ax.plot(epochs, history["train_accuracy"], label="train")
    if history.get("val_accuracy"):
        ax.plot(
            epochs[:len(history["val_accuracy"])],
            history["val_accuracy"],
            label="val",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{dataset} | {mode} — Accuracy")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Artefact: confusion matrix
# ---------------------------------------------------------------------------

def save_confusion_matrix(
    y_true,
    y_pred,
    class_names: list,
    mode: str,
    dataset: str,
):
    os.makedirs("artifacts/confusion_matrices", exist_ok=True)
    path = os.path.join(
        "artifacts", "confusion_matrices", f"{dataset}_{mode}_cm.png"
    )

    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=class_names
    )

    fig, ax = plt.subplots(figsize=(9, 9))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Results serialisation
# ---------------------------------------------------------------------------

def save_results(results: dict, mode: str, dataset: str) -> None:
    os.makedirs("results", exist_ok=True)
    path = f"results/{dataset}_{mode}.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"Saved → {path}")

"""
pipeline.py
===========
End-to-end experiment pipeline.

Pipeline modes
--------------
baseline      CNN features (global avg per video) → MLP classifier
temporal      CNN features + temporal deltas (global avg per video) → MLP
hybrid_svm    CNN + temporal deltas → StandardScaler → LinearSVM
hybrid_elm    CNN + temporal deltas → StandardScaler → ELM
skeleton      MediaPipe pose + velocity → StandardScaler → sklearn MLP
deep_learning CNN + temporal deltas → per-video sequences
              → hyper-parameter search over CNN/LSTM/TCN sequence models

Data splitting
--------------
A single custom stratified split is used consistently across all modes:
  60% train / 20% val / 20% test (per class, with min=1 sample each).
The split is seeded with seed=42 and uses numpy RNG exclusively to avoid
any dependency on sklearn version or its internal random state.

Cache invalidation
------------------
Cache keys encode: dataset, split name (train/val/test), pipeline version,
and a hash of all frame-selection hyper-parameters. Changing any of these
automatically invalidates the cache. This prevents stale features from
a previous run silently corrupting new experiments.

Comparability across modes
--------------------------
All modes share the same:
  - video paths and labels
  - train/val/test split
  - frame extraction parameters (frame_size, num_frames)
  - CNN backbone (MobileNetV2 frozen)
  - frame selection algorithm (for modes that use it)
  - latency measurement protocol (one video at a time, CUDA sync)

This means test accuracies, latencies, and efficiency scores are directly
comparable across rows in the final summary table.
"""

import hashlib
import os
import random
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from features import (
    augment_with_temporal_deltas,
    build_video_sequences,
    combine_features,
    extract_cnn_features,
    load_cached_features,
    save_cached_features,
)
from models import get_model
from preprocessing import (
    MAX_FRAMES_PER_VIDEO,
    FRAME_SELECTION_DIVERSITY_WEIGHT,
    FRAME_SELECTION_MOTION_WEIGHT,
    extract_frames_from_videos,
    extract_skeleton,
    feature_based_frame_selection,
    get_class_list,
    load_video_paths,
)
from utils import (
    evaluate_predictions,
    get_peak_gpu_memory_mb,
    measure_latency,
    reset_gpu_memory_tracking,
    save_confusion_matrix,
    save_results,
    save_training_curve,
)

# Bump this string whenever the pipeline logic changes in a way that
# invalidates previously cached features or results.
PIPELINE_VERSION = "research_v4"

# Frame extraction parameters — part of the cache key.
FRAME_SIZE  = (112, 112)
NUM_FRAMES  = 20  # evenly spaced frames sampled per video before selection


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------

def _frame_selection_hash() -> str:
    """Hash all frame-selection hyper-parameters into an 8-char hex string."""
    params = {
        "max_frames": MAX_FRAMES_PER_VIDEO,
        "div_weight": FRAME_SELECTION_DIVERSITY_WEIGHT,
        "mot_weight": FRAME_SELECTION_MOTION_WEIGHT,
        "num_frames": NUM_FRAMES,
        "frame_size": FRAME_SIZE,
    }
    return hashlib.md5(str(sorted(params.items())).encode()).hexdigest()[:8]


def _cache_key(dataset: str, split: str) -> str:
    return f"{PIPELINE_VERSION}_{dataset}_cnn_{split}_{_frame_selection_hash()}"


def _skeleton_cache_key(dataset: str, split: str) -> str:
    return f"{PIPELINE_VERSION}_{dataset}_skeleton_{split}_{_frame_selection_hash()}"


# ---------------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------------

def split_video_dataset(video_paths, labels, seed: int = 42):
    """Stratified 60/20/20 train/val/test split at the video level.

    Each class contributes at least 1 sample to val and test, regardless of
    class size. The split is performed with a fixed numpy RNG for full
    reproducibility independent of sklearn version.

    Returns
    -------
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)
    """
    rng    = np.random.default_rng(seed)
    labels = np.array(labels)
    indices = np.arange(len(video_paths))

    train_idx, val_idx, test_idx = [], [], []

    for label in np.unique(labels):
        cls_idx = indices[labels == label].copy()
        rng.shuffle(cls_idx)

        n       = len(cls_idx)
        n_test  = max(1, int(round(n * 0.20)))
        n_val   = max(1, int(round(n * 0.20)))

        # Guard: ensure at least 1 training sample per class
        if n_test + n_val >= n:
            n_test = 1
            n_val  = 1

        test_idx.extend(cls_idx[:n_test].tolist())
        val_idx.extend(cls_idx[n_test : n_test + n_val].tolist())
        train_idx.extend(cls_idx[n_test + n_val :].tolist())

    def _take(idxs):
        return [video_paths[i] for i in idxs], labels[idxs]

    return _take(train_idx), _take(val_idx), _take(test_idx)


# ---------------------------------------------------------------------------
# Feature aggregation (frame → video level)
# ---------------------------------------------------------------------------

def aggregate_video_features(
    features: np.ndarray,
    video_ids: np.ndarray,
    frame_labels: np.ndarray,
) -> tuple:
    """Average per-frame features into one vector per video.

    Returns
    -------
    X : float32 (V, D)
    y : int     (V,)
    """
    X, y = [], []

    for vid in sorted(np.unique(video_ids)):
        idxs = np.where(video_ids == vid)[0]
        vid_labels = np.unique(frame_labels[idxs])

        if len(vid_labels) != 1:
            raise ValueError(
                f"Inconsistent frame labels for video {vid}: {vid_labels}"
            )

        X.append(np.mean(features[idxs], axis=0))
        y.append(int(vid_labels[0]))

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_or_compute(key: str, fn):
    cached = load_cached_features(key)
    if cached is not None:
        return cached
    value = fn()
    save_cached_features(key, value)
    return value


def _apply_frame_selection(cnn_feat, frames, labels, video_ids):
    """Run feature_based_frame_selection and return aligned outputs."""
    sel_frames, sel_labels, sel_vids, sel_idx = feature_based_frame_selection(
        cnn_feat, frames, labels, video_ids
    )
    return sel_frames, sel_labels, sel_vids, cnn_feat[sel_idx]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(mode: str, dataset: str = "ucf101") -> dict:
    """Run one experiment and return a results dictionary.

    Parameters
    ----------
    mode    : one of {baseline, temporal, hybrid_svm, hybrid_elm,
                      skeleton, deep_learning}
    dataset : 'ucf101' or 'hmdb51'
    """
    set_seed(42)
    reset_gpu_memory_tracking()
    start_time = time.time()

    print(f"\n{'=' * 60}")
    print(f" mode={mode}  dataset={dataset}  version={PIPELINE_VERSION}")
    print(f"{'=' * 60}\n")

    # ------------------------------------------------------------------
    # 1. Load paths and split
    # ------------------------------------------------------------------
    video_paths, labels = load_video_paths(dataset)
    if len(video_paths) == 0:
        raise RuntimeError(f"No samples found for dataset '{dataset}'.")

    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = \
        split_video_dataset(video_paths, labels)

    print(
        f"Split sizes — train: {len(train_paths)}  "
        f"val: {len(val_paths)}  test: {len(test_paths)}"
    )

    # ------------------------------------------------------------------
    # 2. Frame extraction
    # ------------------------------------------------------------------
    X_train, y_train, vid_train = extract_frames_from_videos(
        train_paths, train_labels,
        frame_size=FRAME_SIZE, num_frames=NUM_FRAMES,
        return_video_ids=True,
    )
    X_val, y_val, vid_val = extract_frames_from_videos(
        val_paths, val_labels,
        frame_size=FRAME_SIZE, num_frames=NUM_FRAMES,
        return_video_ids=True,
    )
    X_test, y_test, vid_test = extract_frames_from_videos(
        test_paths, test_labels,
        frame_size=FRAME_SIZE, num_frames=NUM_FRAMES,
        return_video_ids=True,
    )

    for split_name, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
        if len(X) == 0:
            raise RuntimeError(
                f"Frame extraction produced an empty {split_name} split "
                f"for dataset '{dataset}'."
            )

    class_names   = get_class_list(dataset)
    num_classes   = len(class_names)

    # ------------------------------------------------------------------
    # 3. CNN feature extraction (cached per split)
    # ------------------------------------------------------------------
    train_cnn = _get_or_compute(
        _cache_key(dataset, "train"),
        lambda: extract_cnn_features(X_train),
    )
    val_cnn = _get_or_compute(
        _cache_key(dataset, "val"),
        lambda: extract_cnn_features(X_val),
    )
    test_cnn = _get_or_compute(
        _cache_key(dataset, "test"),
        lambda: extract_cnn_features(X_test),
    )

    # ------------------------------------------------------------------
    # 4. Feature selection (modes that use motion-based frame selection)
    # ------------------------------------------------------------------
    USES_SELECTION = {"temporal", "hybrid_svm", "hybrid_elm", "deep_learning"}

    if mode in USES_SELECTION:
        X_train, y_train, vid_train, train_cnn = _apply_frame_selection(
            train_cnn, X_train, y_train, vid_train
        )
        X_val, y_val, vid_val, val_cnn = _apply_frame_selection(
            val_cnn, X_val, y_val, vid_val
        )
        X_test, y_test, vid_test, test_cnn = _apply_frame_selection(
            test_cnn, X_test, y_test, vid_test
        )

    # ------------------------------------------------------------------
    # 5a. Deep learning mode: sequence-level classification
    # ------------------------------------------------------------------
    if mode == "deep_learning":
        # Temporal delta augmentation
        train_seq = augment_with_temporal_deltas(train_cnn, vid_train)
        val_seq   = augment_with_temporal_deltas(val_cnn,   vid_val)
        test_seq  = augment_with_temporal_deltas(test_cnn,  vid_test)

        # Build fixed-length sequences with padding masks
        X_tr_seq, y_tr_vid, masks_tr = build_video_sequences(train_seq, vid_train, y_train)
        X_v_seq,  y_v_vid,  masks_v  = build_video_sequences(val_seq,   vid_val,   y_val)
        X_te_seq, y_te_vid, masks_te = build_video_sequences(test_seq,  vid_test,  y_test)

        # Per-feature StandardScaler (fitted only on training data)
        # We reshape to (N*T, D), scale, then reshape back.
        scaler = StandardScaler()
        shape_tr = X_tr_seq.shape
        shape_v  = X_v_seq.shape
        shape_te = X_te_seq.shape

        X_tr_seq = scaler.fit_transform(
            X_tr_seq.reshape(-1, shape_tr[-1])
        ).reshape(shape_tr)
        X_v_seq  = scaler.transform(
            X_v_seq.reshape(-1, shape_v[-1])
        ).reshape(shape_v)
        X_te_seq = scaler.transform(
            X_te_seq.reshape(-1, shape_te[-1])
        ).reshape(shape_te)

        seq_dim = X_tr_seq.shape[-1]
        model   = get_model(mode, input_dim=seq_dim, num_classes=num_classes)

        model.fit(
            X_tr_seq, y_tr_vid, X_val=X_v_seq, y_val=y_v_vid,
            masks=masks_tr, masks_val=masks_v,
        )

        val_preds  = model.predict(X_v_seq,  masks_v)
        test_preds = model.predict(X_te_seq, masks_te)

        val_acc  = evaluate_predictions(y_v_vid,  val_preds,  "VAL")
        test_acc = evaluate_predictions(y_te_vid, test_preds, "TEST")

        latency, fps = measure_latency(model, X_te_seq, masks=masks_te)

        cm_path = save_confusion_matrix(
            y_te_vid, test_preds, class_names, mode, dataset
        )

        extra = {
            "feature_source":   "cnn_mobilenetv2_delta_augmented_sequences",
            "best_dl_model":    model.best_model_name_,
            "best_dl_config":   model.best_config_,
            "search_results":   model.search_results_,
        }

    # ------------------------------------------------------------------
    # 5b. All other modes: video-level (aggregated feature) classification
    # ------------------------------------------------------------------
    else:
        if mode == "baseline":
            train_feat = train_cnn
            val_feat   = val_cnn
            test_feat  = test_cnn

        elif mode in ("temporal", "hybrid_svm", "hybrid_elm"):
            train_feat = augment_with_temporal_deltas(train_cnn, vid_train)
            val_feat   = augment_with_temporal_deltas(val_cnn,   vid_val)
            test_feat  = augment_with_temporal_deltas(test_cnn,  vid_test)

        elif mode == "skeleton":
            train_feat = _get_or_compute(
                _skeleton_cache_key(dataset, "train"),
                lambda: augment_with_temporal_deltas(extract_skeleton(X_train), vid_train),
            )
            val_feat = _get_or_compute(
                _skeleton_cache_key(dataset, "val"),
                lambda: augment_with_temporal_deltas(extract_skeleton(X_val), vid_val),
            )
            test_feat = _get_or_compute(
                _skeleton_cache_key(dataset, "test"),
                lambda: augment_with_temporal_deltas(extract_skeleton(X_test), vid_test),
            )

        else:
            raise ValueError(f"Unknown mode: '{mode}'")

        # Aggregate frames → one vector per video
        X_tr_vid, y_tr_vid = aggregate_video_features(train_feat, vid_train, y_train)
        X_v_vid,  y_v_vid  = aggregate_video_features(val_feat,   vid_val,   y_val)
        X_te_vid, y_te_vid = aggregate_video_features(test_feat,  vid_test,  y_test)

        # Scaling for SVM, ELM, skeleton (improves conditioning)
        if mode in ("hybrid_svm", "hybrid_elm", "skeleton"):
            scaler   = StandardScaler()
            X_tr_vid = scaler.fit_transform(X_tr_vid)
            X_v_vid  = scaler.transform(X_v_vid)
            X_te_vid = scaler.transform(X_te_vid)

        # Instantiate and train model
        if mode == "hybrid_svm":
            model = get_model(mode, params={"C": 1})
            model.fit(X_tr_vid, y_tr_vid)
        else:
            model = get_model(mode, X_tr_vid.shape[1], num_classes)
            model.fit(X_tr_vid, y_tr_vid, X_v_vid, y_v_vid)

        val_preds  = model.predict(X_v_vid)
        test_preds = model.predict(X_te_vid)

        val_acc  = evaluate_predictions(y_v_vid,  val_preds,  "VAL")
        test_acc = evaluate_predictions(y_te_vid, test_preds, "TEST")

        latency, fps = measure_latency(model, X_te_vid)

        cm_path = save_confusion_matrix(
            y_te_vid, test_preds, class_names, mode, dataset
        )

        extra = {}
        if mode == "skeleton":
            extra["feature_source"] = "mediapipe_pose_normalised_velocity"

    # ------------------------------------------------------------------
    # 6. Shared result assembly
    # ------------------------------------------------------------------
    total_runtime   = time.time() - start_time
    curve_path      = save_training_curve(
        getattr(model, "history_", {}), mode, dataset
    )
    peak_memory_mb  = get_peak_gpu_memory_mb()
    efficiency      = float(test_acc / latency) if latency > 0 else 0.0

    results = {
        "experiment_version":    PIPELINE_VERSION,
        "mode":                  mode,
        "dataset":               dataset,
        "val_accuracy":          float(val_acc),
        "test_accuracy":         float(test_acc),
        "latency_ms":            float(latency),
        "fps":                   float(fps),
        "efficiency_score":      efficiency,
        "total_runtime_sec":     float(total_runtime),
        "peak_gpu_memory_mb":    float(peak_memory_mb),
        "confusion_matrix_path": cm_path,
        "training_curve_path":   curve_path,
        **extra,
    }

    save_results(results, mode, dataset)

    print(f"\n{'─' * 50}")
    print(f"  mode:             {mode}")
    print(f"  dataset:          {dataset}")
    print(f"  val  accuracy:    {val_acc:.4f}")
    print(f"  test accuracy:    {test_acc:.4f}")
    print(f"  latency (ms):     {latency:.3f}")
    print(f"  FPS:              {fps:.2f}")
    print(f"  efficiency score: {efficiency:.6f}")
    print(f"  peak GPU (MB):    {peak_memory_mb:.2f}")
    print(f"  runtime (s):      {total_runtime:.1f}")
    print(f"{'─' * 50}\n")

    return results

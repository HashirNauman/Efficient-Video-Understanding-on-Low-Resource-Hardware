import hashlib
import random
import time
import warnings

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from features import (
    combine_features,
    extract_cnn_features,
    load_cached_features,
    save_cached_features,
)
from models import get_model
from preprocessing import (
    extract_frames_from_videos,
    extract_skeleton,
    feature_based_frame_selection,
    load_video_paths,
)
from utils import evaluate, measure_latency, save_results


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_config_hash(config_dict):
    return hashlib.md5(str(sorted(config_dict.items())).encode()).hexdigest()[:8]


def aggregate_features(features, video_ids, labels):
    video_dict = {}

    for feature, vid, label in zip(features, video_ids, labels):
        if vid not in video_dict:
            video_dict[vid] = {"features": [], "label": label}
        video_dict[vid]["features"].append(feature)

    X, y = [], []

    for vid in sorted(video_dict.keys()):
        X.append(np.mean(video_dict[vid]["features"], axis=0))
        y.append(video_dict[vid]["label"])

    return np.array(X), np.array(y)


def safe_extract_skeleton(frames, strict=False):
    try:
        return extract_skeleton(frames)
    except ImportError as exc:
        if strict:
            raise
        warnings.warn(f"MediaPipe unavailable, using zero skeleton features: {exc}")
    except Exception as exc:
        if strict:
            raise
        warnings.warn(f"MediaPipe failed, using zero skeleton features: {exc}")

    return np.zeros((len(frames), 99), dtype=np.float32)


def run_pipeline(mode, dataset="ucf101"):
    set_seed(42)
    start_time = time.time()

    print(f"\nRunning mode={mode} dataset={dataset}")

    config_hash = get_config_hash({"mode": mode, "dataset": dataset})
    video_paths, labels = load_video_paths(dataset)

    if len(video_paths) == 0:
        raise RuntimeError(f"No samples found for dataset '{dataset}'")

    X_temp, X_test, y_temp, y_test = train_test_split(
        video_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
    )

    X_train, y_train, vid_train = extract_frames_from_videos(
        X_train, y_train, return_video_ids=True
    )
    X_val, y_val, vid_val = extract_frames_from_videos(
        X_val, y_val, return_video_ids=True
    )
    X_test, y_test, vid_test = extract_frames_from_videos(
        X_test, y_test, return_video_ids=True
    )

    if min(len(X_train), len(X_val), len(X_test)) == 0:
        raise RuntimeError(f"Frame extraction produced an empty split for dataset '{dataset}'")

    train_cnn = extract_cnn_features(X_train)

    if mode in ["temporal", "hybrid_svm", "hybrid_elm", "multimodal"]:
        X_train, y_train, vid_train, idx = feature_based_frame_selection(
            train_cnn, X_train, y_train, vid_train
        )
        train_cnn = train_cnn[idx]

    def get_features(X, cnn, split):
        key = f"{dataset}_{mode}_{split}_{config_hash}"
        cached = load_cached_features(key)
        if cached is not None:
            return cached

        if mode == "multimodal":
            skeleton_features = safe_extract_skeleton(X, strict=False)
            feat = combine_features(cnn, skeleton_features)
        elif mode == "skeleton":
            feat = safe_extract_skeleton(X, strict=True)
        else:
            feat = cnn

        save_cached_features(key, feat)
        return feat

    val_cnn = extract_cnn_features(X_val)
    test_cnn = extract_cnn_features(X_test)

    train_feat = get_features(X_train, train_cnn, "train")
    val_feat = get_features(X_val, val_cnn, "val")
    test_feat = get_features(X_test, test_cnn, "test")

    train_feat, y_train = aggregate_features(train_feat, vid_train, y_train)
    val_feat, y_val = aggregate_features(val_feat, vid_val, y_val)
    test_feat, y_test = aggregate_features(test_feat, vid_test, y_test)

    if mode in ["hybrid_svm", "hybrid_elm", "multimodal"]:
        scaler = StandardScaler()
        train_feat = scaler.fit_transform(train_feat)
        val_feat = scaler.transform(val_feat)
        test_feat = scaler.transform(test_feat)

    if mode in ["hybrid_svm", "multimodal"]:
        model = get_model(mode, params={"C": 1})
        model.fit(train_feat, y_train)
    else:
        model = get_model(mode, train_feat.shape[1], len(np.unique(labels)))
        model.fit(train_feat, y_train)

    val_acc = evaluate(model, val_feat, y_val, split_name="VAL")
    test_acc = evaluate(model, test_feat, y_test, split_name="TEST")
    latency, fps = measure_latency(model, test_feat)
    total_runtime = time.time() - start_time

    results = {
        "mode": mode,
        "dataset": dataset,
        "val_accuracy": float(val_acc),
        "test_accuracy": float(test_acc),
        "latency_ms": float(latency),
        "fps": float(fps),
        "efficiency_score": float(test_acc / latency) if latency > 0 else 0.0,
        "total_runtime_sec": float(total_runtime),
    }

    save_results(results, mode, dataset)

    print("Run Summary")
    print(f"mode: {mode}")
    print(f"dataset: {dataset}")
    print(f"accuracy: {test_acc:.4f}")
    print(f"latency (ms): {latency:.3f}")
    print(f"FPS: {fps:.2f}")
    print(f"efficiency score: {results['efficiency_score']:.6f}")
    print(f"total runtime: {total_runtime:.2f} sec")

    return results

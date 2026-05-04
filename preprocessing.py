"""
preprocessing.py
================
Frame loading, skeleton extraction, and feature-based frame selection.

Design decisions documented for reproducibility:
- Pose landmarks are pelvis-centred and shoulder-width-normalised (scale-invariant).
- Velocity (frame-to-frame delta) is appended to every pose feature vector.
- Frame selection uses a deterministic greedy algorithm: pick the highest-motion
  frame first, then greedily add the frame maximising a weighted combination of
  diversity (cosine distance from already-chosen set) and motion magnitude.
  Hard cap: max_frames_per_video frames are retained per video so that
  build_video_sequences can stack them into a fixed-shape tensor.
"""

import os
import warnings

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances


MEDIAPIPE_IMPORT_ERROR = None

try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
except Exception as exc:
    mp_pose = None
    MEDIAPIPE_IMPORT_ERROR = exc


# ---------------------------------------------------------------------------
# Class lists
# ---------------------------------------------------------------------------

UCF101_CLASSES = [
    "BoxingPunchingBag", "Punch", "SumoWrestling", "SkateBoarding",
    "SoccerPenalty", "Basketball", "CricketShot", "TennisSwing",
    "JumpRope", "WalkingWithDog", "ShavingBeard", "CuttingInKitchen",
    "WritingOnBoard", "Typing", "PlayingFlute",
]

HMDB51_CLASSES = [
    "punch", "kick", "run", "jump", "climb", "shake_hands", "hug",
    "push", "pullup", "ride_bike", "sit", "stand", "smile", "talk", "eat",
]

# Number of frames kept per video by feature_based_frame_selection.
# All downstream sequence models rely on this being constant.
MAX_FRAMES_PER_VIDEO = 8

# Weight balancing diversity vs motion in frame selection.
FRAME_SELECTION_DIVERSITY_WEIGHT = 0.7
FRAME_SELECTION_MOTION_WEIGHT = 0.3


def get_class_list(dataset: str) -> list:
    return UCF101_CLASSES if dataset == "ucf101" else HMDB51_CLASSES


# ---------------------------------------------------------------------------
# Video / frame-folder loading
# ---------------------------------------------------------------------------

def load_video_paths(dataset: str = "ucf101"):
    """Return (video_paths, labels) for the requested dataset.

    UCF101: paths are individual video files listed in data/ucf101/train.csv.
    HMDB51: paths are per-video frame directories under data/hmdb51/<class>/.
    """
    base_path = f"data/{dataset}"
    video_paths: list = []
    labels: list = []

    classes = get_class_list(dataset)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    if dataset == "ucf101":
        split_file = f"{base_path}/train.csv"
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Missing split file: {split_file}")

        df = pd.read_csv(split_file)
        skipped_labels: dict = {}

        path_column = None
        for candidate in ["video_path", "clip_path", "path"]:
            if candidate in df.columns:
                path_column = candidate
                break

        if path_column is None:
            raise KeyError(
                f"Expected one of ['video_path', 'clip_path', 'path'] in "
                f"{split_file}, found {list(df.columns)}"
            )

        for _, row in df.iterrows():
            label_name = row["label"]
            if label_name not in class_to_idx:
                skipped_labels[label_name] = skipped_labels.get(label_name, 0) + 1
                continue

            relative_path = str(row[path_column]).lstrip("/\\")
            video_paths.append(os.path.join(base_path, relative_path))
            labels.append(class_to_idx[label_name])

        if skipped_labels:
            warnings.warn(
                f"Filtered UCF101 to the configured {len(UCF101_CLASSES)} classes. "
                f"Skipped {sum(skipped_labels.values())} samples from "
                f"{len(skipped_labels)} other classes."
            )

        if not video_paths:
            raise RuntimeError(
                "No UCF101 samples matched the configured UCF101_CLASSES list."
            )

        return video_paths, np.array(labels)

    # HMDB51: folder-based
    for cls in classes:
        class_dir = os.path.join(base_path, cls)
        if not os.path.exists(class_dir):
            continue

        for vid_folder in os.listdir(class_dir):
            vid_path = os.path.join(class_dir, vid_folder)
            if os.path.isdir(vid_path):
                video_paths.append(vid_path)
                labels.append(class_to_idx[cls])

    if not video_paths:
        raise RuntimeError(
            f"No HMDB51 samples found under {base_path}. "
            "Check that class folder names match HMDB51_CLASSES."
        )

    return video_paths, np.array(labels)


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frames_from_videos(
    video_paths,
    labels,
    frame_size: tuple = (112, 112),
    num_frames: int = 20,
    return_video_ids: bool = False,
):
    """Decode videos / frame directories and sample `num_frames` evenly spaced
    frames per video.

    Returns
    -------
    frames        : float32 array of shape (N, H, W, 3), values in [0, 1]
    frame_labels  : int array of shape (N,)
    video_ids     : int array of shape (N,)  [only when return_video_ids=True]
    """
    frames, frame_labels, video_ids = [], [], []

    for vid_id, (path, label) in enumerate(zip(video_paths, labels)):
        video_frames = []

        try:
            if path.endswith((".mp4", ".avi", ".mkv")):
                if not os.path.exists(path):
                    warnings.warn(f"Skipping missing video file: {path}")
                    continue

                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    warnings.warn(f"Skipping unreadable video file: {path}")
                    continue

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.resize(frame, frame_size) / 255.0
                    video_frames.append(frame)

                cap.release()
            else:
                if not os.path.isdir(path):
                    warnings.warn(f"Skipping missing frame directory: {path}")
                    continue

                frame_files = sorted(os.listdir(path))
                for frame_name in frame_files:
                    img_path = os.path.join(path, frame_name)
                    frame = cv2.imread(img_path)
                    if frame is None:
                        warnings.warn(f"Skipping corrupted frame: {img_path}")
                        continue
                    frame = cv2.resize(frame, frame_size) / 255.0
                    video_frames.append(frame)

        except Exception as exc:
            warnings.warn(f"Skipping corrupted sample {path}: {exc}")
            continue

        if len(video_frames) == 0:
            warnings.warn(f"Skipping sample with no usable frames: {path}")
            continue

        idxs = np.linspace(0, len(video_frames) - 1, num_frames).astype(int)
        for i in idxs:
            frames.append(video_frames[i])
            frame_labels.append(label)
            video_ids.append(vid_id)

    if return_video_ids:
        return (
            np.array(frames, dtype=np.float32),
            np.array(frame_labels),
            np.array(video_ids),
        )

    return np.array(frames, dtype=np.float32), np.array(frame_labels)


# ---------------------------------------------------------------------------
# Skeleton extraction
# ---------------------------------------------------------------------------

def _normalize_pose_landmarks(landmarks) -> np.ndarray:
    """Pelvis-centre and shoulder-width normalisation.

    This makes the pose representation invariant to absolute position and
    overall body scale while preserving relative joint angles and proportions.

    Output: flattened float32 vector of length 99 (33 joints × 3 coords).
    """
    coords = np.array(
        [[p.x, p.y, p.z] for p in landmarks],
        dtype=np.float32,
    )
    # Pelvis = midpoint of left hip (23) and right hip (24)
    pelvis = (coords[23] + coords[24]) / 2.0
    centered = coords - pelvis
    # Scale by shoulder width (left=11, right=12); add epsilon to avoid /0
    scale = np.linalg.norm(coords[11] - coords[12]) + 1e-6
    return (centered / scale).reshape(-1)


def extract_skeleton(frames: np.ndarray) -> np.ndarray:
    """Run MediaPipe Pose on every frame and return a (N, 198) feature matrix.

    Each row = [normalised_pose (99 dims) | velocity (99 dims)].
    Velocity is zero for the first frame of each call.

    Raises ImportError if MediaPipe is not installed.
    """
    if mp_pose is None:
        raise ImportError(f"MediaPipe Pose unavailable: {MEDIAPIPE_IMPORT_ERROR}")

    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=0,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    )
    features = []
    prev = None

    for frame in frames:
        try:
            img = (frame * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = pose.process(img)

            if result.pose_landmarks:
                current = _normalize_pose_landmarks(result.pose_landmarks.landmark)
            else:
                current = np.zeros(99, dtype=np.float32)
        except Exception as exc:
            warnings.warn(f"MediaPipe failed on frame, using zeros: {exc}")
            current = np.zeros(99, dtype=np.float32)

        velocity = np.zeros_like(current) if prev is None else current - prev
        features.append(np.concatenate([current, velocity]).astype(np.float32))
        prev = current

    pose.close()
    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Feature-based frame selection
# ---------------------------------------------------------------------------

def feature_based_frame_selection(
    features: np.ndarray,
    frames: np.ndarray,
    labels: np.ndarray,
    video_ids: np.ndarray,
    max_frames_per_video: int = MAX_FRAMES_PER_VIDEO,
) -> tuple:
    """Select a fixed number of informative frames per video.

    Algorithm
    ---------
    1. Seed with the frame of highest L2 motion magnitude.
    2. Greedily add frames maximising:
       score = DIVERSITY_WEIGHT * min_cosine_distance_to_chosen_set
             + MOTION_WEIGHT   * normalised_motion_magnitude
    3. Sort chosen indices temporally before returning.

    This guarantees every video contributes exactly `max_frames_per_video`
    frames (or all its frames if it has fewer), enabling downstream
    `np.stack` into a fixed-shape sequence tensor.

    Returns
    -------
    sel_frames, sel_labels, sel_video_ids, selected_global_indices
    """
    selected_idx = []

    for vid in np.unique(video_ids):
        idxs = np.where(video_ids == vid)[0]

        if len(idxs) <= max_frames_per_video:
            # Keep all frames; pad is handled by build_video_sequences
            selected_idx.extend(idxs.tolist())
            continue

        vid_features = features[idxs]

        # Motion magnitude (L2 of inter-frame delta)
        motion = np.zeros(len(idxs), dtype=np.float32)
        if len(idxs) > 1:
            motion[1:] = np.linalg.norm(np.diff(vid_features, axis=0), axis=1)

        local_chosen = [int(np.argmax(motion))]
        all_local = list(range(len(idxs)))

        while len(local_chosen) < max_frames_per_video:
            remaining = [i for i in all_local if i not in local_chosen]

            dist_scores = cosine_distances(
                vid_features[remaining],
                vid_features[local_chosen],
            ).min(axis=1)

            motion_remaining = motion[remaining]
            motion_max = np.max(motion_remaining)
            norm_motion = (
                motion_remaining / motion_max if motion_max > 0
                else motion_remaining
            )

            combined = (
                FRAME_SELECTION_DIVERSITY_WEIGHT * dist_scores
                + FRAME_SELECTION_MOTION_WEIGHT * norm_motion
            )
            local_chosen.append(int(remaining[int(np.argmax(combined))]))

        local_chosen = sorted(local_chosen)
        selected_idx.extend(idxs[local_chosen].tolist())

    selected_idx = np.array(selected_idx)
    return (
        frames[selected_idx],
        labels[selected_idx],
        video_ids[selected_idx],
        selected_idx,
    )

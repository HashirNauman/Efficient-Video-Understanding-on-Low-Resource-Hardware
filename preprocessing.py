import os
import cv2
import numpy as np
import pandas as pd
import warnings

try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
except:
    mp_pose = None


# =========================================================
# CLASS LISTS
# =========================================================
UCF101_CLASSES = [
    "BoxingPunchingBag","Punch","SumoWrestling","SkateBoarding",
    "SoccerPenalty","Basketball","CricketShot","TennisSwing",
    "JumpRope","WalkingWithDog","ShavingBeard","CuttingInKitchen",
    "WritingOnBoard","Typing","PlayingFlute"
]

HMDB51_CLASSES = [
    "punch","kick","run","jump","climb","shake_hands","hug",
    "push","pullup","ride_bike","sit","stand","smile","talk","eat"
]


def get_class_list(dataset):
    return UCF101_CLASSES if dataset == "ucf101" else HMDB51_CLASSES


# =========================================================
# STEP 1: LOAD VIDEO / FRAME PATHS (FIXED FOR BOTH DATASETS)
# =========================================================
def load_video_paths(dataset="ucf101"):
    base_path = f"data/{dataset}"

    video_paths = []
    labels = []

    classes = get_class_list(dataset)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # =========================
    # UCF101 (CSV-based splits)
    # =========================
    if dataset == "ucf101":
        split_file = f"{base_path}/train.csv"
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Missing split file: {split_file}")
        df = pd.read_csv(split_file)

        # expected columns: [video_path, label]
        for _, row in df.iterrows():
            video_paths.append(os.path.join(base_path, row["video_path"]))
            labels.append(class_to_idx[row["label"]])

        return video_paths, np.array(labels)

    # =========================
    # HMDB51 (frame folders)
    # =========================
    else:
        for cls in classes:
            class_dir = os.path.join(base_path, cls)
            if not os.path.exists(class_dir):
                continue

            for vid_folder in os.listdir(class_dir):
                vid_path = os.path.join(class_dir, vid_folder)

                if os.path.isdir(vid_path):
                    video_paths.append(vid_path)
                    labels.append(class_to_idx[cls])

        return video_paths, np.array(labels)


# =========================================================
# STEP 2: FRAME EXTRACTION (FIXED FOR BOTH FORMATS)
# =========================================================
def extract_frames_from_videos(video_paths, labels, frame_size=(112,112), num_frames=20, return_video_ids=False):

    frames, frame_labels, video_ids = [], [], []

    for vid_id, (path, label) in enumerate(zip(video_paths, labels)):

        video_frames = []

        try:
            # =========================
            # CASE 1: VIDEO FILE (UCF101)
            # =========================
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

            # =========================
            # CASE 2: FRAME FOLDER (HMDB51)
            # =========================
            else:
                if not os.path.isdir(path):
                    warnings.warn(f"Skipping missing frame directory: {path}")
                    continue

                frame_files = sorted(os.listdir(path))

                for f in frame_files:
                    img_path = os.path.join(path, f)
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

        idxs = np.linspace(0, len(video_frames)-1, num_frames).astype(int)

        for i in idxs:
            frames.append(video_frames[i])
            frame_labels.append(label)
            video_ids.append(vid_id)

    if return_video_ids:
        return np.array(frames), np.array(frame_labels), np.array(video_ids)

    return np.array(frames), np.array(frame_labels)


# =========================================================
# TEMPORAL FILTER
# =========================================================
def filter_frames(frames, labels, video_ids, threshold=0.95):
    if len(frames) == 0:
        return frames, labels, video_ids

    f_frames, f_labels, f_vids = [frames[0]], [labels[0]], [video_ids[0]]

    for i in range(1, len(frames)):
        diff = np.mean(np.abs(frames[i] - frames[i - 1]))
        if diff > (1 - threshold):
            f_frames.append(frames[i])
            f_labels.append(labels[i])
            f_vids.append(video_ids[i])

    return np.array(f_frames), np.array(f_labels), np.array(f_vids)


# =========================================================
# SKELETON EXTRACTION
# =========================================================
def extract_skeleton(frames):
    if mp_pose is None:
        raise ImportError("Install mediapipe")

    pose = mp_pose.Pose(static_image_mode=True)
    features = []

    for frame in frames:
        try:
            img = (frame * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = pose.process(img)

            if result.pose_landmarks:
                lm = []
                for p in result.pose_landmarks.landmark:
                    lm.extend([p.x, p.y, p.z])
            else:
                lm = [0] * 99
        except Exception as exc:
            warnings.warn(f"MediaPipe failed on one frame, using zeros: {exc}")
            lm = [0] * 99

        features.append(lm)

    return np.array(features, dtype=np.float32)


# =========================================================
# FRAME SELECTION (FIXED CLEAN VERSION)
# =========================================================
from sklearn.metrics.pairwise import cosine_similarity

def feature_based_frame_selection(features, frames, labels, video_ids, threshold=0.9):

    selected_idx = []

    for vid in np.unique(video_ids):
        idxs = np.where(video_ids == vid)[0]

        prev_feat = None

        for i in idxs:
            feat = features[i].reshape(1, -1)

            if prev_feat is None:
                keep = True
            else:
                sim = cosine_similarity(feat, prev_feat)[0][0]
                keep = sim < threshold

            if keep:
                selected_idx.append(i)
                prev_feat = feat

    selected_idx = np.array(selected_idx)

    return (
        frames[selected_idx],
        labels[selected_idx],
        video_ids[selected_idx],
        selected_idx
    )

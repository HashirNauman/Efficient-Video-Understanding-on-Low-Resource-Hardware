# Efficient Video Understanding on Low-Resource Hardware

A benchmark comparing ML classifiers (SVM, ELM) and deep sequence models (1-D CNN, TCN, Bi-LSTM) for action recognition on a commodity GPU, with motion-driven keyframe selection to reduce temporal redundancy.

---

## Overview

Most video understanding research assumes high-end server GPUs. This project asks a practical question: **can lightweight ML classifiers on top of frozen CNN features match deep sequence models, while running faster on a mid-range laptop GPU?**

Findings on a Quadro M2000M (4 GB VRAM, 2016 mobile workstation):

| Model | UCF101 Acc | HMDB51 Acc | Latency (ms) |
|---|---|---|---|
| Baseline (CNN → MLP) | 80.8% | 55.7% | 0.55 |
| Temporal (CNN + deltas → MLP) | 83.2% | 55.7% | 0.62 |
| Hybrid SVM | **92.9%** | **68.1%** | 1.85 |
| Hybrid ELM | 76.4% | 46.3% | 0.99 |
| Deep learning (NAS over CNN/TCN/LSTM) | **94.4%** | **68.9%** | 5.24 |

Key result: hybrid SVM achieves within 1.5% of the best deep learning model on UCF101 at **2.8× lower inference latency**, making it a strong candidate for latency-critical deployment on constrained hardware.

---

## Pipeline

```
Videos / frame folders
        │
        ▼
Frame extraction (20 evenly spaced frames, 112×112)
        │
        ▼
MobileNetV2 feature extraction (frozen, ImageNet weights → 1280-d per frame)
        │
        ├── Baseline / Temporal ──► Global average pool ──► MLP
        │
        ├── Hybrid SVM / ELM ──► Frame selection (motion + diversity) ──► Temporal deltas ──► StandardScaler ──► SVM / ELM
        │
        └── Deep learning ──► Frame selection ──► Temporal deltas ──► Sequences ──► NAS (CNN / TCN / Bi-LSTM)
```

### Frame selection algorithm

For modes that use it (`temporal`, `hybrid_svm`, `hybrid_elm`, `deep_learning`):
1. Seed with the frame of highest L2 motion magnitude.
2. Greedily add frames maximising `0.7 × cosine_diversity + 0.3 × normalised_motion`.
3. Cap at 8 frames per video and sort temporally.

This is deterministic given the same seed and produces a fixed-length sequence for downstream models.

---

## Datasets

**UCF101** (15-class subset): video files from the UCF101 action recognition dataset. Place in `data/ucf101/` with a `train.csv` listing `video_path` and `label` columns.

**HMDB51** (15-class subset): frame directories under `data/hmdb51/<class>/<video>/`. Frames should be JPEG or PNG images sorted by filename.

Classes used:

```python
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
```

---

## Installation

```bash
pip install torch torchvision opencv-python scikit-learn pandas matplotlib 
```


---

## Running experiments

```bash
python experiment_runner.py
```

This runs all five experiments (baseline, temporal, hybrid_svm, hybrid_elm, deep_learning) on every available dataset and saves results to `results/` and `results_summary/`.

To run a single experiment:

```python
from pipeline import run_pipeline
results = run_pipeline(mode="hybrid_svm", dataset="ucf101")
```

Valid modes: `baseline`, `temporal`, `hybrid_svm`, `hybrid_elm`, `skeleton`, `deep_learning`.

---

## Output structure

```
results/
    ucf101_baseline.json
    ucf101_hybrid_svm.json
    ...
results_summary/
    all_experiments.csv          ← scalar metrics, all experiments
    all_experiments_full.csv     ← includes NAS search results
artifacts/
    confusion_matrices/
    training_curves/
feature_cache/                   ← cached CNN features (auto-invalidated on pipeline change)
```

---

## Reproducibility

- All experiments use a fixed seed (42) for Python, NumPy, and PyTorch.
- Train/val/test split is 60/20/20 stratified per class, implemented with `numpy.random.default_rng(42)` — independent of sklearn version.
- CNN feature caches are keyed on pipeline version + frame-selection hyperparameters. Changing any parameter automatically invalidates the cache.
- `PIPELINE_VERSION = "research_v4"` in `pipeline.py` — bump this string to force a full re-run.

---

## Hardware

All experiments ran on:

- GPU: NVIDIA Quadro M2000M (4 GB VRAM, Maxwell architecture)
- CPU: Intel Core i7 (laptop)
- OS: Windows 10

Latency is measured per video (one sample at a time), with a 5-sample CUDA warm-up before the timed window, and `torch.cuda.synchronize()` before and after. This makes latency directly comparable across all modes.

---

## Project structure

```
preprocessing.py      — frame loading, skeleton extraction, frame selection
features.py           — CNN extraction, temporal deltas, sequence building, caching
models.py             — CNN_Softmax, ELM, MLP, TemporalConvNet, SequenceCNN, SequenceLSTM, NAS search
utils.py              — evaluation, latency measurement, confusion matrix, training curves
pipeline.py           — end-to-end experiment runner for a single mode
experiment_runner.py  — runs all experiments, saves summary CSV
```

---

## Limitations

- 15-class subsets only. Results are not directly comparable to full UCF101 (101 classes) or HMDB51 (51 classes) benchmarks.
- Each experiment was run once with seed=42. Multiple independent runs with different seeds are needed to establish statistical significance.

---

## Acknowledgements

Datasets: UCF101 (Soomro et al., 2012), HMDB51 (Kuehne et al., 2011).  
Backbone: MobileNetV2 (Sandler et al., 2018), pretrained on ImageNet.  

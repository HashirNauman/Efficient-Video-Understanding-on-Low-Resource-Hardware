"""
experiment_runner.py
====================
Runs the active benchmark experiments across all available datasets.

Experiment set
--------------
baseline      - CNN frame features, MLP classifier
temporal      - CNN + temporal deltas, MLP classifier
hybrid_svm    - CNN + deltas, LinearSVM
hybrid_elm    - CNN + deltas, Extreme Learning Machine
deep_learning - CNN + deltas organised as video sequences,
                model search over {SequenceCNN, SequenceLSTM, TCN}
"""

import json
import os
import time
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd
import torch

from pipeline import PIPELINE_VERSION, run_pipeline


EXPERIMENTS = [
    {"name": "baseline", "mode": "baseline"},
    {"name": "temporal", "mode": "temporal"},
    {"name": "hybrid_svm", "mode": "hybrid_svm"},
    {"name": "hybrid_elm", "mode": "hybrid_elm"},
    {"name": "deep_learning", "mode": "deep_learning"},
]

DATASETS = ["ucf101", "hmdb51"]


def _result_path(dataset: str, mode: str) -> str:
    return os.path.join("results", f"{dataset}_{mode}.json")


def _load_existing(dataset: str, mode: str):
    path = _result_path(dataset, mode)
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            result = json.load(f)
    except Exception as exc:
        warnings.warn(f"Cannot read result file {path}: {exc}")
        return None

    if result.get("dataset") != dataset or result.get("mode") != mode:
        warnings.warn(f"Ignoring mismatched result file: {path}")
        return None

    if result.get("experiment_version") != PIPELINE_VERSION:
        print(
            f"[SKIP] Stale result for {dataset}/{mode} "
            f"(found {result.get('experiment_version')}, need {PIPELINE_VERSION})"
        )
        return None

    return result


def _validate_dataset(dataset):
    dataset_root = os.path.join("data", dataset)
    if not os.path.exists(dataset_root):
        return False, f"Missing dataset root: {dataset_root}"

    if dataset == "ucf101":
        required_items = [
            os.path.join(dataset_root, "train.csv"),
            os.path.join(dataset_root, "train"),
        ]
    else:
        required_items = [dataset_root]

    missing_items = [path for path in required_items if not os.path.exists(path)]
    if missing_items:
        warnings.warn(
            f"Dataset '{dataset}' is partially missing. Missing: {missing_items}. "
            "Continuing because the root exists."
        )

    return True, f"Dataset ready: {dataset_root}"


def _available_datasets():
    datasets = []
    for dataset in DATASETS:
        ok, message = _validate_dataset(dataset)
        print(message)
        if ok:
            datasets.append(dataset)
    return datasets


def _print_device_status():
    print("Using GPU" if torch.cuda.is_available() else "Using CPU (slow execution expected)")


def run_all():
    datasets = _available_datasets()
    if not datasets:
        raise SystemExit("No valid datasets found. Stopping execution immediately.")

    os.makedirs("results", exist_ok=True)
    os.makedirs("results_summary", exist_ok=True)
    _print_device_status()

    summary = []
    failures = []
    expected = len(datasets) * len(EXPERIMENTS)

    for dataset in datasets:
        print(f"\n==============================")
        print(f"DATASET: {dataset}")
        print(f"==============================")

        for exp in EXPERIMENTS:
            mode = exp["mode"]
            existing = _load_existing(dataset, mode)
            if existing is not None:
                print(f"\nSkipping completed experiment: {mode}")
                existing.setdefault("runner_runtime_sec", None)
                summary.append(existing)
                continue

            run_start = time.time()
            print(f"\nRunning Experiment: {mode}")

            try:
                result = run_pipeline(mode=mode, dataset=dataset)
            except Exception as exc:
                failures.append({"dataset": dataset, "mode": mode, "error": str(exc)})
                warnings.warn(f"Experiment failed for {dataset}/{mode}: {exc}")
                continue

            result["runner_runtime_sec"] = float(time.time() - run_start)
            summary.append(result)

    if not summary:
        raise RuntimeError("No experiments completed successfully.")

    df = pd.DataFrame(summary)
    df = df.sort_values(["dataset", "mode"]).reset_index(drop=True)

    csv_path = "results_summary/all_experiments.csv"
    df_display = df.copy()
    df_display.to_csv(csv_path, index=False)

    full_csv = "results_summary/all_experiments_full.csv"
    df.to_csv(full_csv, index=False)

    print("\nFINAL SUMMARY TABLE")
    print(df_display)

    n_done = len(summary)
    if n_done == expected:
        print(f"\nAll {expected} experiments completed successfully.")
    else:
        print(f"\nCompleted {n_done}/{expected} experiments.")

    print(f"Summary saved -> {csv_path}")
    print(f"Full summary saved -> {full_csv}")

    if failures:
        print("\nFailed experiments:")
        for failure in failures:
            print(f"  {failure['dataset']}/{failure['mode']}: {failure['error']}")


if __name__ == "__main__":
    run_all()

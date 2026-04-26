import os
import time
import warnings

import pandas as pd
import torch

from pipeline import run_pipeline


EXPERIMENTS = [
    {"name": "baseline", "mode": "baseline"},
    {"name": "temporal", "mode": "temporal"},
    {"name": "hybrid_svm", "mode": "hybrid_svm"},
    {"name": "hybrid_elm", "mode": "hybrid_elm"},
    {"name": "skeleton", "mode": "skeleton"},
    {"name": "multimodal", "mode": "multimodal"},
]

DATASETS = ["ucf101", "hmdb51"]


def validate_dataset(dataset):
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


def get_available_datasets():
    available = []

    for dataset in DATASETS:
        ok, message = validate_dataset(dataset)
        print(message)
        if ok:
            available.append(dataset)

    return available


def print_device_status():
    using_gpu = torch.cuda.is_available()
    print("Using GPU" if using_gpu else "Using CPU (slow execution expected)")


def run_all():
    available_datasets = get_available_datasets()
    if not available_datasets:
        raise SystemExit("No valid datasets found. Stopping execution immediately.")

    os.makedirs("results_summary", exist_ok=True)
    print_device_status()

    summary = []

    for dataset in available_datasets:
        print(f"\n==============================")
        print(f"DATASET: {dataset}")
        print(f"==============================")

        for exp in EXPERIMENTS:
            run_start = time.time()
            print(f"\nRunning Experiment: {exp['name']}")

            try:
                results = run_pipeline(mode=exp["mode"], dataset=dataset)
            except Exception as exc:
                warnings.warn(
                    f"Experiment failed for mode={exp['mode']} dataset={dataset}: {exc}"
                )
                continue

            results["runner_runtime_sec"] = float(time.time() - run_start)
            summary.append(results)

    if not summary:
        raise RuntimeError("No experiments completed successfully.")

    df = pd.DataFrame(summary)
    csv_path = "results_summary/all_experiments.csv"
    df.to_csv(csv_path, index=False)

    print("\nFINAL SUMMARY TABLE")
    print(df)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("SUMMARY SAVED AT: results_summary/all_experiments.csv")


if __name__ == "__main__":
    run_all()

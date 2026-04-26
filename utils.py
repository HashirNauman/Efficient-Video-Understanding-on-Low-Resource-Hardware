import json
import os
import time

import numpy as np


def evaluate(model, X, y, split_name="TEST"):
    print(f"Evaluating on {split_name} set...")

    preds = model.predict(X)
    accuracy = np.mean(preds == y)

    print(f"{split_name} Accuracy: {accuracy:.4f}")
    return accuracy


def measure_latency(model, X, num_samples=100):
    print("Measuring latency...")

    X_sample = X[:min(num_samples, len(X))]
    if len(X_sample) == 0:
        return 0.0, 0.0

    start = time.time()

    for i in range(len(X_sample)):
        _ = model.predict(X_sample[i:i + 1])

    end = time.time()
    total_time = end - start

    if total_time <= 0:
        return 0.0, 0.0

    latency = (total_time / len(X_sample)) * 1000
    fps = len(X_sample) / total_time

    print(f"Latency: {latency:.3f} ms | FPS: {fps:.2f}")
    return latency, fps


def save_results(results, mode, dataset):
    os.makedirs("results", exist_ok=True)

    filename = f"results/{dataset}_{mode}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"Saved -> {filename}")

import argparse
import csv
from pathlib import Path
import yaml

import matplotlib.pyplot as plt
import numpy as np

from experiments.utils import plot_mean_std, average_runs_with_std


def average_metric(histories, metric_key):
    if not histories:
        return []
    steps = len(histories[0])
    n = len(histories)
    denom = max(1, n - 1)
    averaged = []
    for i in range(steps):
        if "acquired" in histories[0][i]:
            acquired = int(histories[0][i]["acquired"])
        else:
            acquired = int(histories[0][i].get("labels", 0))
        
        if "acquired_from_pool" in histories[0][i]:
            acquired_from_pool = int(histories[0][i]["acquired_from_pool"])
        else:
            acquired_from_pool = int(histories[0][i].get("labels_from_pool", acquired))
        
        vals = [float(h[i][metric_key]) for h in histories]
        mean_v = sum(vals) / n
        std_v = (sum((v - mean_v) ** 2 for v in vals) / denom) ** 0.5
        entry = {
            "step": i,
            metric_key: mean_v,
            f"{metric_key}_std": std_v,
        }
        if "acquired" in histories[0][i]:
            entry["acquired"] = acquired
        if "labels" in histories[0][i]:
            entry["labels"] = int(histories[0][i].get("labels", acquired))
        if "acquired_from_pool" in histories[0][i]:
            entry["acquired_from_pool"] = acquired_from_pool
        if "labels_from_pool" in histories[0][i]:
            entry["labels_from_pool"] = int(histories[0][i].get("labels_from_pool", acquired_from_pool))
        averaged.append(entry)
    return averaged


def load_csv(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        r["step"] = int(r["step"])
        if "acquired" in r:
            r["acquired"] = int(r["acquired"])
        if "labels" in r:
            r["labels"] = int(r["labels"])
        if "acquired_from_pool" in r:
            r["acquired_from_pool"] = int(r["acquired_from_pool"])
        if "labels_from_pool" in r:
            r["labels_from_pool"] = int(r["labels_from_pool"])
        for key in ("test_accuracy", "test_loss", "test_rmse"):
            if key in r:
                r[key] = float(r[key])
        out.append(r)
    return out


def plot_classification(eta_dir, acqs, repeats):
    curves = {}
    for acq in acqs:
        runs = []
        for attempt in range(repeats):
            path = eta_dir / f"class_{acq}_attempt_{attempt}.csv"
            if not path.exists():
                continue
            runs.append(load_csv(path))
        if runs:
            curves[acq] = average_runs_with_std(runs)
    if not curves:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    label_map = {"bald": "BALD", "entropy": "Predictive Entropy", "random": "Random"}
    for name, hist in curves.items():
        x_key = "acquired_from_pool" if "acquired_from_pool" in hist[0] else "acquired"
        x = np.array([h[x_key] for h in hist], dtype=float)
        y = np.array([h["test_accuracy"] for h in hist], dtype=float) * 100.0
        s = np.array([h.get("test_accuracy_std", 0.0) for h in hist], dtype=float) * 100.0
        plot_mean_std(ax, x, y, s, label=label_map.get(name, name))
    ax.set(xlabel="Number of acquired points", ylabel="Test accuracy (%)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    
    folder_name = f"{eta_dir.parent.name}_{eta_dir.name}"
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    out_path = plots_dir / f"{folder_name}_classification_plot.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    return out_path


def plot_regression(eta_dir, inferences, acqs, repeats):
    curves = {}
    for inference in inferences:
        for acq in acqs:
            runs = []
            for attempt in range(repeats):
                path = eta_dir / f"reg_{inference}_{acq}_attempt_{attempt}.csv"
                if not path.exists():
                    continue
                runs.append(load_csv(path))
            if runs:
                key = f"{inference}:{acq}"
                curves[key] = average_metric(runs, "test_rmse")
    if not curves:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, hist in curves.items():
        label = name.replace(":", " / ")
        x_key = "labels_from_pool" if "labels_from_pool" in hist[0] else ("acquired_from_pool" if "acquired_from_pool" in hist[0] else "labels")
        x = np.array([h[x_key] for h in hist], dtype=float)
        y = np.array([h["test_rmse"] for h in hist], dtype=float)
        s = np.array([h.get("test_rmse_std", 0.0) for h in hist], dtype=float)
        plot_mean_std(ax, x, y, s, label=label)
    ax.set(
        xlabel="Number of acquired points",
        ylabel="Test RMSE (clean test set)",
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    
    folder_name = f"{eta_dir.parent.name}_{eta_dir.name}"
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    out_path = plots_dir / f"{folder_name}_regression_plot.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/noisy_oracle.yaml")
    parser.add_argument("--results-directory", type=str, default="results/noisy_oracle")
    args = parser.parse_args()

    with open(args.config) as f:
        noise_cfg = yaml.safe_load(f)

    results_dir = Path(args.results_directory)

    saved = []
    for eta in noise_cfg["noise_val"]:
        eta_dir = results_dir / f"eta_{eta}"
        if not eta_dir.exists():
            continue
        cls_path = plot_classification(eta_dir, noise_cfg["class_af"], noise_cfg["repeats"])
        reg_path = plot_regression(eta_dir, noise_cfg["regression_inference"], noise_cfg["regression_af"], noise_cfg["repeats"])
        if cls_path:
            saved.append(cls_path)
        if reg_path:
            saved.append(reg_path)


if __name__ == "__main__":
    main()


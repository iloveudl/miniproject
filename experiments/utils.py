import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def series(hist, key):
    x = np.array([h["acquired_from_pool"] for h in hist], dtype=float)
    y = 100.0 * np.array([h[key] for h in hist], dtype=float)
    s = 100.0 * np.array([h.get(f"{key}_std", 0.0) for h in hist], dtype=float)
    return x, y, s


def plot_mean_std(ax, x, y, s, label):
    (line,) = ax.plot(x, y, label=label)
    ax.fill_between(x, y - s, y + s, alpha=0.2, color=line.get_color())


def average_runs_with_std(histories):
    if not histories:
        return []
    steps, n = len(histories[0]), len(histories)
    denom = max(1, n - 1)
    averaged = []
    for i in range(steps):
        acquired = histories[0][i]["acquired"]
        acquired_from_pool = histories[0][i]["acquired_from_pool"]
        accuracies = [h[i]["test_accuracy"] for h in histories]
        losses = [h[i]["test_loss"] for h in histories]
        test_accuracy = sum(accuracies) / n
        test_loss = sum(losses) / n
        test_accuracy_std = (sum((a - test_accuracy) ** 2 for a in accuracies) / denom) ** 0.5
        test_loss_std = (sum((v - test_loss) ** 2 for v in losses) / denom) ** 0.5
        averaged.append(
            {
                "step": i,
                "acquired": acquired,
                "test_accuracy": test_accuracy,
                "test_accuracy_std": test_accuracy_std,
                "test_loss": test_loss,
                "test_loss_std": test_loss_std,
                "acquired_from_pool": acquired_from_pool,
            }
        )
    return averaged


def save_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def plot_exp1_histories(histories, results_dir, table_rows=None):
    label_map = {
        "bald": "BALD",
        "variation": "Var Ratios",
        "entropy": "Max Entropy",
        "mean_std": "Mean STD",
        "random": "Random",
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    all_ys = []
    for name, hist in histories.items():
        x, y, s = series(hist, "test_accuracy")
        plot_mean_std(ax, x, y, s, label=label_map.get(name, name))
        all_ys.extend(y - s)
        all_ys.extend(y + s)
    
    if all_ys:
        y_min = max(0, min(all_ys) - 2)
        y_max = min(100, max(all_ys) + 2)
        ylim = (y_min, y_max)
    else:
        ylim = (80, 100)

    ax.set(
        xlabel="Number of acquired images",
        ylabel="Test accuracy",
        ylim=ylim,
        xlim=(0, max(hist[-1]["acquired_from_pool"] for hist in histories.values())),
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    
    folder_name = results_dir.name if results_dir.name else results_dir.parent.name
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    out_path = plots_dir / f"{folder_name}_exp_1_plot.pdf"
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")

    if table_rows:
        table_path = results_dir / "exp1_table.csv"
        header = "acquisition,threshold,acquired_mean,acquired_std,step_mean,step_std"
        lines = [header]
        for row in table_rows:
            lines.append(
                f"{row['acquisition']},{row['threshold']},{row['acquired_mean']},{row['acquired_std']},{row['step_mean']},{row['step_std']}"
            )
        table_path.write_text("\n".join(lines))


def plot_exp2_histories(bayes_avg, det_avg, results_dir):
    label_map = {"bald": "BALD", "variation": "Var Ratios", "entropy": "Max Entropy"}
    acquisitions = list(bayes_avg)

    fig, axes = plt.subplots(1, len(acquisitions), figsize=(5 * len(acquisitions), 4), sharey=False)
    axes = np.atleast_1d(axes)

    for ax, acq in zip(axes, acquisitions):
        x_b, y_b, s_b = series(bayes_avg[acq], "test_accuracy")
        x_d, y_d, s_d = series(det_avg[acq], "test_accuracy")

        title = label_map.get(acq, acq)
        plot_mean_std(ax, x_b, y_b, s_b, label=title)
        plot_mean_std(ax, x_d, y_d, s_d, label=f"Deterministic {title}")

        all_y_values = list(y_b - s_b) + list(y_b + s_b) + list(y_d - s_d) + list(y_d + s_d)
        if all_y_values:
            y_min = max(0, min(all_y_values) - 2)
            y_max = min(100, max(all_y_values) + 2)
            ax.set_ylim(y_min, y_max)
        else:
            ax.set_ylim(80, 100)

        ax.set_xlabel("Number of added images")
        ax.set_xlim(0, 1000)
        ax.set_xticks(list(range(0, 1001, 100)))  
        ax.tick_params(labelbottom=True, labelleft=True)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="lower right")

    axes[0].set_ylabel("Test accuracy")
    fig.tight_layout()

    folder_name = results_dir.name if results_dir.name else results_dir.parent.name
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    out_path = plots_dir / f"{folder_name}_exp2_plot.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")

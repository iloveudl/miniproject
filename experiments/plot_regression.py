import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import csv
import yaml


def average_runs(runs):
    if not runs:
        return [], None
    steps = len(runs[0])
    n = len(runs)
    denom = max(1, n - 1)
    out = []
    upper_bound_rmse = None
    has_ood = False
    if runs and runs[0]:
        first_row = runs[0][0]
        if "pool_ood_ratio" in first_row:
            try:
                pool_ratio = float(first_row.get("pool_ood_ratio", 0.0) or 0.0)
                has_ood = pool_ratio > 1e-6
            except (ValueError, TypeError):
                has_ood = False
    
    for i in range(steps):
        acquired = runs[0][i].get("labels", runs[0][i].get("acquired", 0))
        labels_key = "labels_from_pool" if "labels_from_pool" in runs[0][i] else "acquired_from_pool"
        labels = runs[0][i].get(labels_key, acquired)
        rmses = [r[i]["test_rmse"] for r in runs]
        mean_rmse = sum(rmses) / n
        std_rmse = (sum((v - mean_rmse) ** 2 for v in rmses) / denom) ** 0.5
        if upper_bound_rmse is None and runs[0][i].get("upper_bound_rmse") is not None:
            upper_bound_rmse = runs[0][i]["upper_bound_rmse"]
        entry = {
            "step": i,
            "acquired": acquired,
            "labels": labels,
            "acquired_from_pool": runs[0][i].get("acquired_from_pool", labels),
            "test_rmse": mean_rmse,
            "test_rmse_std": std_rmse,
        }
        if has_ood:
            ood_step = [float(r[i].get("ood_in_batch_rate", 0.0) or 0.0) for r in runs]
            ood_cum = [float(r[i].get("ood_cum_rate", 0.0) or 0.0) for r in runs]
            entry["ood_in_batch_rate"] = sum(ood_step) / n
            entry["ood_in_batch_rate_std"] = (sum((v - entry["ood_in_batch_rate"]) ** 2 for v in ood_step) / denom) ** 0.5
            entry["ood_cum_rate"] = sum(ood_cum) / n
            entry["ood_cum_rate_std"] = (sum((v - entry["ood_cum_rate"]) ** 2 for v in ood_cum) / denom) ** 0.5
        out.append(entry)
    return out, upper_bound_rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base_regression.yaml")
    parser.add_argument("--results-directory", type=str, default="results/regression")
    parser.add_argument( "--methods", type=str, default="analytic, mfvi, vi, vi_full")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    acq_mode_default = str(cfg.get("regression_af", "trace")).strip().lower()

    results_dir = Path(args.results_directory)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    is_ood_experiment = bool(str(cfg.get("ood_regression_dataset", "")).strip().lower())

    if is_ood_experiment:
        folder_name = results_dir.name if results_dir.name else results_dir.parent.name
        expected_ratio = None
        expected_ratios = [0.0, 0.1, 0.3, 0.5]
        
        import re
        if folder_name.endswith("_0") or folder_name.endswith("_0.0"):
            expected_ratio = 0.0
        else:
            for r in expected_ratios:
                if r == 0.0:
                    continue 
                pattern1 = f"_{r}"
                pattern2 = f"_{int(r * 10)}"
                if pattern1 in folder_name or folder_name.endswith(pattern1) or pattern2 in folder_name or folder_name.endswith(pattern2):
                    expected_ratio = r
                    break
            
            if expected_ratio is None:
                match = re.search(r'[0_]+([135])', folder_name.replace('.', '_'))
                if match:
                    digit = match.group(1)
                    if digit == '1':
                        expected_ratio = 0.1
                    elif digit == '3':
                        expected_ratio = 0.3
                    elif digit == '5':
                        expected_ratio = 0.5
        
        def parse_filename(csv_file):
            parts = csv_file.stem.split("_")
            if len(parts) >= 3 and parts[-2] == "attempt":
                if len(parts) >= 4 and parts[0] == "vi" and parts[1] == "full":
                    method_part = "_".join(parts[0:2])
                    acq_part = parts[2]
                else:
                    method_part = parts[0]
                    acq_part = "_".join(parts[1:-2]) if len(parts) > 3 else parts[1]
                return method_part, acq_part
            return None, None
        
        all_acqs = set()
        for csv_file in results_dir.glob("*.csv"):
            method_part, acq_part = parse_filename(csv_file)
            if method_part and method_part in methods:
                all_acqs.add(acq_part)
        
        # results are stored flat under results_dir (no per-method subfolders)
        
        if not all_acqs:
            raise ValueError(f"No CSV files found in {results_dir}. Run experiments first.")
        
        all_acqs = sorted(all_acqs)
        
        data_by_method_acq = defaultdict(lambda: defaultdict(list))
        
        for method in methods:
            for acq in all_acqs:
                runs = []
                for attempt in range(args.repeats):
                    path = results_dir / f"{method}_{acq}_attempt_{attempt}.csv"
                    if not path.exists():
                        continue
                    with path.open(newline="") as f:
                        rows = list(csv.DictReader(f))
                    
                    if rows and "pool_ood_ratio" in rows[0] and expected_ratio is not None:
                        first_ratio_val = float(rows[0]["pool_ood_ratio"])
                        rounded_first = min(expected_ratios, key=lambda x: abs(x - first_ratio_val))
                        if abs(rounded_first - expected_ratio) > 1e-6:
                            continue
                    
                    filtered_rows = []
                    for r in rows:
                        r["step"] = int(r["step"])
                        if "labels" in r:
                            r["labels"] = int(r["labels"])
                        if "acquired" in r:
                            r["acquired"] = int(r["acquired"])
                        if "labels_from_pool" in r:
                            r["labels_from_pool"] = int(r["labels_from_pool"])
                        if "acquired_from_pool" in r:
                            r["acquired_from_pool"] = int(r["acquired_from_pool"])
                        r["test_rmse"] = float(r["test_rmse"])
                        if "pool_ood_ratio" in r:
                            ratio_val = float(r["pool_ood_ratio"])
                            rounded_ratio = min(expected_ratios, key=lambda x: abs(x - ratio_val))
                            r["pool_ood_ratio"] = rounded_ratio
                            
                            if expected_ratio is not None and abs(rounded_ratio - expected_ratio) > 1e-6:
                                continue
                        if "ood_in_batch_rate" in r:
                            r["ood_in_batch_rate"] = float(r["ood_in_batch_rate"])
                        if "ood_cum_rate" in r:
                            r["ood_cum_rate"] = float(r["ood_cum_rate"])
                        filtered_rows.append(r)
                    
                    if filtered_rows:
                        runs.append(filtered_rows)
                
                if runs:
                    first_ratio = runs[0][0].get("pool_ood_ratio", 0.0)
                    if expected_ratio is None:
                        expected_ratio = first_ratio
                    data_by_method_acq[method][acq].extend(runs)
        
        if not data_by_method_acq:
            raise ValueError(f"No data found matching expected ratio {expected_ratio} in {results_dir}.")
        
        if expected_ratio is None:
            first_method = next(iter(data_by_method_acq.keys()))
            first_acq = next(iter(data_by_method_acq[first_method].keys()))
            first_file = data_by_method_acq[first_method][first_acq][0][0]
            expected_ratio = float(first_file.get("pool_ood_ratio", 0.0))
        
        ratio = expected_ratio
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
        all_y_rmse = []
        all_y_ood = []
        
        label_map = {
            "analytic": "Analytic",
            "mfvi": "MFVI",
            "vi": "vi",
            "vi_full": "vi-full",
        }
        
        for method in methods:
            if method not in data_by_method_acq:
                continue
            for acq in all_acqs:
                if acq not in data_by_method_acq[method]:
                    continue
                runs = data_by_method_acq[method][acq]
                hist, ub_rmse = average_runs(runs)
                if not hist:
                    continue

                if "labels_from_pool" in hist[0]:
                    x_key = "labels_from_pool"
                elif "labels" in hist[0]:
                    x_key = "labels"
                else:
                    x_key = "acquired_from_pool"
                x = np.array([h[x_key] for h in hist], dtype=float)
                y_rmse = np.array([h["test_rmse"] for h in hist], dtype=float)
                y_ood = np.array([h.get("ood_cum_rate", 0.0) for h in hist], dtype=float)
                
                method_label = label_map.get(method, method)
                label = f"{method_label} / {acq}"
                line1, = ax1.plot(x, y_rmse, label=label)
                line2, = ax2.plot(x, y_ood, label=label, color=line1.get_color())
                
                all_y_rmse.extend(y_rmse)
                all_y_ood.extend(y_ood)
        
        if all_y_rmse:
            y_min_rmse = max(0, min(all_y_rmse) - 0.02 * (max(all_y_rmse) - min(all_y_rmse)))
            y_max_rmse = max(all_y_rmse) + 0.02 * (max(all_y_rmse) - min(all_y_rmse))
            ax1.set_ylim(y_min_rmse, y_max_rmse)
        
        if all_y_ood:
            y_min_ood = max(0, min(all_y_ood) - 0.05)
            y_max_ood = min(1, max(all_y_ood) + 0.05)
            ax2.set_ylim(y_min_ood, y_max_ood)
        
        ax1.set_ylabel("Test RMSE (ID test set)")
        ax1.grid(True, linestyle="--", alpha=0.4)
        ax1.legend()
        ax1.tick_params(labelbottom=True, labelleft=True)
        
        ax2.set_xlabel("Number of acquired labels from pool")
        ax2.set_ylabel("OOD selection rate (cumulative)")
        ax2.grid(True, linestyle="--", alpha=0.4)
        ax2.legend()
        ax2.tick_params(labelbottom=True, labelleft=True)
        
        plt.tight_layout()
        
        folder_name = results_dir.name if results_dir.name else results_dir.parent.name
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        
        ratio_str = str(ratio).replace(".", "_")
        out_path = plots_dir / f"{folder_name}_regression_ood_ratio{ratio_str}.pdf"
        plt.savefig(out_path, format="pdf", bbox_inches="tight")
    else:
        curves = {}
        upper_bounds = {}
        random_curve = None
        random_upper = None
        
        random_runs = []
        for attempt in range(args.repeats):
            path = results_dir / f"analytic_random_attempt_{attempt}.csv"
            if not path.exists():
                continue
            with path.open(newline="") as f:
                rows = list(csv.DictReader(f))
            for r in rows:
                r["step"] = int(r["step"])
                if "labels" in r:
                    r["labels"] = int(r["labels"])
                if "acquired" in r:
                    r["acquired"] = int(r["acquired"])
                if "labels_from_pool" in r:
                    r["labels_from_pool"] = int(r["labels_from_pool"])
                if "acquired_from_pool" in r:
                    r["acquired_from_pool"] = int(r["acquired_from_pool"])
                r["test_rmse"] = float(r["test_rmse"])
            if rows:
                random_runs.append(rows)
        if random_runs:
            random_hist, random_upper = average_runs(random_runs)
            random_curve = random_hist
        
        for method in methods:
            runs = []
            for attempt in range(args.repeats):
                path = None
                test_path = results_dir / f"{method}_{acq_mode_default}_attempt_{attempt}.csv"
                if test_path.exists():
                    path = test_path
                
                if path is None or not path.exists():
                    continue  # Skip this attempt if file doesn't exist
                with path.open(newline="") as f:
                    rows = list(csv.DictReader(f))
                for r in rows:
                    r["step"] = int(r["step"])
                    if "labels" in r:
                        r["labels"] = int(r["labels"])
                    if "acquired" in r:
                        r["acquired"] = int(r["acquired"])
                    if "labels_from_pool" in r:
                        r["labels_from_pool"] = int(r["labels_from_pool"])
                    if "acquired_from_pool" in r:
                        r["acquired_from_pool"] = int(r["acquired_from_pool"])
                    r["test_rmse"] = float(r["test_rmse"])
                runs.append(rows)
            if runs:  # Only add if we found at least one run
                hist, upper_bound_rmse = average_runs(runs)
                curves[method] = hist
                if upper_bound_rmse is not None:
                    upper_bounds[method] = upper_bound_rmse

        is_diversity = bool(cfg.get("diversity_lambda", 0.0)) or "div" in str(results_dir).lower()
        diversity_suffix = " (diversity)" if is_diversity else ""
        
        label_map = {
            "analytic": "Analytic inference",
            "mfvi": "Mean-field VI",
            "vi": "Matrix-normal VI (diag-U full-V)",
            "vi_full": "Matrix-normal VI (full-U full-V)",
        }

        fig, ax = plt.subplots(figsize=(7, 4))
        all_y_values = []
        
        if random_curve:
            x_key = "labels" if "labels" in random_curve[0] else "acquired_from_pool"
            x = np.array([h[x_key] for h in random_curve], dtype=float)
            y = np.array([h["test_rmse"] for h in random_curve], dtype=float)
            s = np.array([h["test_rmse_std"] for h in random_curve], dtype=float)
            label = "Random" + diversity_suffix
            line, = ax.plot(x, y, label=label, color="gray", linestyle="-", linewidth=1.5)
            ax.fill_between(x, y - s, y + s, alpha=0.2, color=line.get_color())
            all_y_values.extend(y - s)
            all_y_values.extend(y + s)
        
        for method, hist in curves.items():
            x_key = "labels" if "labels" in hist[0] else "acquired_from_pool"
            x = np.array([h[x_key] for h in hist], dtype=float)
            y = np.array([h["test_rmse"] for h in hist], dtype=float)
            s = np.array([h["test_rmse_std"] for h in hist], dtype=float)
            label = label_map.get(method, method) + diversity_suffix
            line, = ax.plot(x, y, label=label)
            ax.fill_between(x, y - s, y + s, alpha=0.2, color=line.get_color())
            all_y_values.extend(y - s)
            all_y_values.extend(y + s)

        if all_y_values:
            y_min = max(0, min(all_y_values) - 0.02 * (max(all_y_values) - min(all_y_values)))
            y_max = max(all_y_values) + 0.02 * (max(all_y_values) - min(all_y_values))
            ax.set_ylim(y_min, y_max)
        
        ax.set_xlabel("Number of labels")
        ax.set_ylabel("Test RMSE")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        ax.tick_params(labelbottom=True, labelleft=True)
        plt.tight_layout()
        
        folder_name = results_dir.name if results_dir.name else results_dir.parent.name
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        
        out_path = plots_dir / f"{folder_name}_regression_plot.pdf"
        plt.savefig(out_path, format="pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()

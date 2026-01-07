import argparse
import csv
from pathlib import Path

from .utils import plot_exp1_histories, average_runs_with_std


def threshold_table_rows(per_attempt, acquisition):
    def mean_std(vals):
        if len(vals) == 1:
            return vals[0], 0.0
        m = sum(vals) / len(vals)
        s = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
        return m, s

    rows = []
    for thr in (0.05, 0.10):
        hits = []
        for run in per_attempt:
            idx = next((i for i, h in enumerate(run) if (1.0 - h["test_accuracy"]) <= thr), None)
            if idx is not None:
                hits.append((run[idx]["acquired_from_pool"], idx))

        if not hits:
            rows.append(
                {
                    "acquisition": acquisition,
                    "threshold": thr,
                    "acquired_mean": "NA",
                    "acquired_std": "NA",
                    "step_mean": "NA",
                    "step_std": "NA",
                }
            )
            continue

        acq_vals, step_vals = zip(*hits)
        mean_acquired, std_acquired = mean_std(acq_vals)
        mean_step, std_step = mean_std(step_vals)
        rows.append(
            {
                "acquisition": acquisition,
                "threshold": thr,
                "acquired_mean": mean_acquired,
                "acquired_std": std_acquired,
                "step_mean": mean_step,
                "step_std": std_step,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-directory", type=str, default="results")
    parser.add_argument("--acquisitions", type=str, default="bald, variation, entropy, mean_std, random")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    res_dir = Path(args.results_directory)
    acquisitions = [s.strip().lower() for s in args.acquisitions.split(",") if s.strip()]

    histories_avg = {}
    table_rows = []
    for acquisition in acquisitions:
        per_attempt = []
        for attempt in range(args.repeats):
            path = Path(res_dir / f"{acquisition}_bayesian_attempt_{attempt}.csv")
            if not path.exists():
                continue

            with path.open(newline="") as f:
                rows = list(csv.DictReader(f))

            for r in rows:
                r["step"] = int(r["step"])
                r["acquired"] = int(r["acquired"])
                r["acquired_from_pool"] = int(r["acquired_from_pool"])
                r["test_accuracy"] = float(r["test_accuracy"])
                r["test_loss"] = float(r["test_loss"])
                r["attempt"] = int(r["attempt"])

            per_attempt.append(rows)

        if per_attempt:
            histories_avg[acquisition] = average_runs_with_std(per_attempt)
            table_rows.extend(threshold_table_rows(per_attempt, acquisition))

    if not histories_avg:
        raise FileNotFoundError(f"No CSV files found in {res_dir}. Run experiments/run_cnns.py")

    plot_exp1_histories(histories_avg, res_dir, table_rows=table_rows)


if __name__ == "__main__":
    main()

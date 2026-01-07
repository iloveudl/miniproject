import argparse
import csv
from pathlib import Path

from .utils import plot_exp2_histories, average_runs_with_std

def load_runs(results_dir, acquisition, regime, repeats):
    runs = []
    for attempt in range(repeats):
        if regime == "bayes":
            path = results_dir / f"{acquisition}_bayesian_attempt_{attempt}.csv"
        elif regime == "det":
            path = results_dir / f"{acquisition}_deterministic_attempt_{attempt}.csv"
        else:
            raise ValueError(f"Unknown regime: {regime}")
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
        runs.append(rows)
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-directory", type=str, default="results")
    parser.add_argument("--acquisitions", type=str, default="bald, variation, entropy")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    results_dir = Path(args.results_directory)
    acquisitions = [s.strip().lower() for s in args.acquisitions.split(",") if s.strip()]

    bayes_avg = {}
    det_avg = {}
    for acquisition in acquisitions:
        bayes_runs = load_runs(results_dir, acquisition, "bayes", args.repeats)
        det_runs = load_runs(results_dir, acquisition, "det", args.repeats)
        if bayes_runs:
            bayes_avg[acquisition] = average_runs_with_std(bayes_runs)
        if det_runs:
            det_avg[acquisition] = average_runs_with_std(det_runs)

    if not bayes_avg and not det_avg:
        raise FileNotFoundError(f"No CSV files found in {results_dir}")

    plot_exp2_histories(bayes_avg, det_avg, results_dir)


if __name__ == "__main__":
    main()

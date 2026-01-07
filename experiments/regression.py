import argparse
from pathlib import Path
import yaml

from experiments.utils import save_csv
from regression_active_process import run_active_process_regression


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base_regression.yaml")
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--results-directory", type=str, default="results/regression")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label-noise", type=float, default=0.0)
    parser.add_argument("--regression-af", type=str, default=None)
    parser.add_argument("--pool-ood-ratio", type=float, default=None)
    args = parser.parse_args()


    res_dir = Path(args.results_directory)
    res_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        base_config = yaml.safe_load(f)

    if args.regression_af:
        acquisitions = [args.regression_af]
    else:
        acquisitions = ["trace", "random"]

    for inference in ["analytic", "mfvi", "vi", "vi_full"]:
        for acq_mode in acquisitions:
            for attempt in range(args.repeats):
                cfg = base_config.copy()

                cfg["seed"] = args.seed + attempt
                cfg["data"] = args.data
                cfg["inference_method"] = inference
                cfg["regression_af"] = acq_mode
                cfg["label_noise"] = float(args.label_noise)
                if args.pool_ood_ratio is not None:
                    cfg["pool_ood_ratio"] = float(args.pool_ood_ratio)

                out_csv = res_dir / f"{inference}_{acq_mode}_attempt_{attempt}.csv"
                if out_csv.exists():
                    continue

                hist = run_active_process_regression(argparse.Namespace(**cfg))
                save_csv(out_csv, hist)


if __name__ == "__main__":
    main()


import argparse
from pathlib import Path
import yaml

from experiments.utils import save_csv
from active_process import run_active_process
from regression_active_process import run_active_process_regression


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/noisy_oracle.yaml")
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--results-directory", type=str, default="results/noisy_oracle")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-classification", action="store_true")
    parser.add_argument("--skip-regression", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        noise_cfg = yaml.safe_load(f)

    res_dir = Path(args.results_directory)
    res_dir.mkdir(parents=True, exist_ok=True)

    for eta in noise_cfg["noise_val"]:
        eta_dir = res_dir / f"eta_{eta}"
        eta_dir.mkdir(parents=True, exist_ok=True)

        if not args.skip_classification:
            for acq in noise_cfg["class_af"]:
                for attempt in range(noise_cfg["repeats"]):
                    cfg = noise_cfg.copy()
                    cfg["seed"] = args.seed + attempt
                    cfg["data"] = args.data
                    cfg["acquisition_function"] = acq
                    cfg["label_noise"] = float(eta)

                    out_csv = eta_dir / f"class_{acq}_attempt_{attempt}.csv"
                    if out_csv.exists():
                        continue
                    hist = run_active_process(argparse.Namespace(**cfg))
                    save_csv(out_csv, hist)

        if not args.skip_regression:
            for inference in noise_cfg["regression_inference"]:
                for acq in noise_cfg["regression_af"]:
                    for attempt in range(noise_cfg["repeats"]):
                        cfg = noise_cfg.copy()
                        cfg["seed"] = args.seed + attempt
                        cfg["data"] = args.data
                        cfg["inference_method"] = inference
                        cfg["regression_af"] = acq
                        cfg["label_noise"] = float(eta)

                        out_csv = eta_dir / f"reg_{inference}_{acq}_attempt_{attempt}.csv"
                        if out_csv.exists():
                            continue
                        hist = run_active_process_regression(argparse.Namespace(**cfg))
                        save_csv(out_csv, hist)


if __name__ == "__main__":
    main()


import argparse
from pathlib import Path
import yaml

from experiments.utils import save_csv
from active_process import run_active_process


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--results-directory", type=str, default="results")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label-noise", type=float, default=0.0)
    args = parser.parse_args()

    res_dir = Path(args.results_directory)
    res_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        base_config = yaml.safe_load(f)

    for acquisition in ["bald", "variation", "entropy", "mean_std", "random"]:
        for attempt in range(args.repeats):
            config_dict = base_config.copy()

            config_dict["seed"] = args.seed + attempt
            config_dict["data"] = args.data
            config_dict["acquisition_function"] = acquisition
            config_dict["label_noise"] = float(args.label_noise)

            out_csv = res_dir / f"{acquisition}_bayesian_attempt_{attempt}.csv"
            if not out_csv.exists():
                history = run_active_process(argparse.Namespace(**config_dict))
                save_csv(out_csv, history)

    for acquisition in ["bald", "variation", "entropy"]:
        for attempt in range(args.repeats):
            config_dict = base_config.copy()

            config_dict["seed"] = args.seed + attempt
            config_dict["data"] = args.data
            config_dict["acquisition_function"] = acquisition
            config_dict["t_pool"] = 1
            config_dict["t_test"] = 1
            config_dict["use_dropout_in_acquisition"] = False
            config_dict["use_dropout_in_eval"] = False
            config_dict["label_noise"] = float(args.label_noise)

            out_csv = res_dir / f"{acquisition}_deterministic_attempt_{attempt}.csv"
            if not out_csv.exists():
                history = run_active_process(argparse.Namespace(**config_dict))
                save_csv(out_csv, history)


if __name__ == "__main__":
    main()

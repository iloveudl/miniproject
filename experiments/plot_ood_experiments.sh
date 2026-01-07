#!/bin/bash

cd "$(dirname "$0")/.." || exit 1

CONFIG="configs/ood_mnist_fashion.yaml"
BASE_DIR="results/ood/regression_ood"

for ratio in 0.0 0.1 0.3 0.5; do
    echo "Plotting with pool_ood_ratio=$ratio..."
    python3 -m experiments.plot_regression \
        --config "$CONFIG" \
        --results-directory "${BASE_DIR}_${ratio}"
done

echo "Done"


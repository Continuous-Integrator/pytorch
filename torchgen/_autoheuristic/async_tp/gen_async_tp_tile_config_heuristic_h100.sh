#!/bin/bash
# Generate the AsyncTPTileConfigH100 heuristic artifact from existing training data.
# Run from: torchgen/_autoheuristic/async_tp/
#
# Prerequisites:
#   1. Collect benchmark data:
#      cd /tmp && PYTHONPATH=~/fbsource/genai/msl \
#          torchrun --nproc_per_node=8 \
#          ~/fbsource/genai/msl/ops/benchmarks/dist_gemm/bench_tile_configs.py \
#          --sweep_m --output_dir /tmp/tile_config_sweep_results
#
#   2. Convert to AH format:
#      python convert_tile_config_data.py \
#          --csv /tmp/tile_config_sweep_results/tile_config_feasibility.csv \
#          --output async_tp_tile_config_h100_data.txt
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DATA_FILE="async_tp_tile_config_h100_data.txt"

if [ ! -f "$DATA_FILE" ]; then
    echo "Data file not found: $DATA_FILE"
    echo "Run: python convert_tile_config_data.py --csv <path-to-csv> --output $DATA_FILE"
    exit 1
fi

python train_decision_async_tp_tile_config.py \
    "${DATA_FILE}" \
    --heuristic-name AsyncTPTileConfigH100

echo "Generated artifact: torch/_inductor/autoheuristic/artifacts/_AsyncTPTileConfigH100.py"

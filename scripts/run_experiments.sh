#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=$(python -c "from src.utils import load_config; print(load_config('configs/experiments/common.yaml')['comparison']['output'])")
rm -rf "$OUT/runs" "$OUT/tables" "$OUT"/figures/0*.png
mkdir -p "$OUT"
exec > >(tee "$OUT/run_experiments.log") 2>&1
mapfile -t CONFIGS < <(python -m src.registry --list)
python -m src.data --config configs/baseline.yaml
python -m src.features --config "${CONFIGS[0]}"
for cfg in "${CONFIGS[@]}"; do
    echo "=== $cfg"
    python -m src.train --config "$cfg"
done
python -m src.compare

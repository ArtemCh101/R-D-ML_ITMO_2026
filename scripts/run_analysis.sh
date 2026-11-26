#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=$(python -c "from src.utils import load_config; print(load_config('configs/analysis/hw3.yaml')['output'])")
rm -rf "$OUT/tables" "$OUT"/figures/0*.png
mkdir -p "$OUT"
exec > >(tee "$OUT/analysis.log") 2>&1
python -m src.data --config configs/baseline.yaml > /dev/null
python -m src.registry > /dev/null
python -m src.error_analysis
python -m src.robustness

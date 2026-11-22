#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mapfile -t CONFIGS < <(python -m src.registry --list)
python -m src.data --config configs/baseline.yaml
python -m src.features --config "${CONFIGS[0]}"
for cfg in "${CONFIGS[@]}"; do
    echo "=== $cfg"
    python -m src.train --config "$cfg"
    python -m src.evaluate --config "$cfg"
done
python -m src.compare

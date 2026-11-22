#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/baseline.yaml}"
python -m src.data --config "$CONFIG"
python -m src.train --config "$CONFIG"
python -m src.evaluate --config "$CONFIG"

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -c "from src.utils import load_config; import yaml; assert load_config('configs/baseline.yaml') == yaml.safe_load(open('configs/baseline.yaml')); print('baseline config loads unchanged')"
python -m src.registry
python -m src.tracking --selftest
python -m src.compare --selftest

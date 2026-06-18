#!/bin/bash

set -e

echo "Start M5 FOODS demand forecasting pipeline"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"

python3 -m venv .venv

source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

python src/run_pipeline.py \
  --notebook notebooks/m5_foods_demand_forecasting.ipynb \
  --output-notebook notebooks/executed_m5_foods_demand_forecasting.ipynb

echo "Pipeline finished"
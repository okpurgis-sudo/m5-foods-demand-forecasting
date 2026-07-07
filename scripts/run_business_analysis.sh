#!/bin/bash

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Running M5 FOODS business analysis..."

if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

python src/business_analysis.py

echo "Business analysis finished."
echo "Generated files:"
echo "results/"
ls -lh results | grep business || true
echo "images/"
ls -lh images | grep business || true
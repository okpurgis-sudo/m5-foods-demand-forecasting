#!/bin/bash

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

BUCKET_NAME="m5-foods-forecasting-taskune257"
PREFIX="m5-foods-demand-forecasting"

echo "Uploading results to S3..."
aws s3 sync results "s3://$BUCKET_NAME/$PREFIX/results" \
  --exclude ".gitkeep"

echo "Uploading images to S3..."
aws s3 sync images "s3://$BUCKET_NAME/$PREFIX/images" \
  --exclude ".gitkeep"

echo "Uploading models to S3..."
aws s3 sync models "s3://$BUCKET_NAME/$PREFIX/models" \
  --exclude ".gitkeep"

echo "Upload completed."
echo "S3 contents:"
aws s3 ls "s3://$BUCKET_NAME/$PREFIX/" --recursive --human-readable --summarize

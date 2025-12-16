#!/bin/bash
# 2x OS (6-9m) モデルを時系列予測で再訓練

set -e

echo "=========================================="
echo "2x OS (6-9m) 時系列予測版 訓練開始"
echo "=========================================="

OUTPUT_DIR="outputs/50projects_irl_timeseries/2x_os"

uv run python scripts/train/train_cross_temporal_multiproject.py \
  --reviews data/openstack_50proj_2021_2024_feat.csv \
  --output "$OUTPUT_DIR" \
  --negative-oversample-factor 2 \
  --epochs 50 \
  --learning-rate 0.001 \
  --min-history-events 3

echo "=========================================="
echo "訓練完了"
echo "出力ディレクトリ: $OUTPUT_DIR"
echo "=========================================="

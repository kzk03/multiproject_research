#!/bin/bash
# 2x OSモデル（最良性能）から特徴量を抽出

set -e

echo "========================================"
echo "2x OSモデルから14次元特徴量を抽出"
echo "========================================"
echo ""

MODEL="outputs/50projects_irl/2x_os/train_6-9m/irl_model.pt"
DATA="data/openstack_50proj_2021_2024_feat.csv"
OUTPUT="outputs/analysis_data/developer_state_features_2x_6-9m.csv"

if [ ! -f "$MODEL" ]; then
    echo "Error: モデルファイルが見つかりません: $MODEL"
    exit 1
fi

if [ ! -f "$DATA" ]; then
    echo "Error: データファイルが見つかりません: $DATA"
    exit 1
fi

echo "モデル: $MODEL"
echo "データ: $DATA"
echo "出力: $OUTPUT"
echo ""

uv run python scripts/analysis/extract_state_features.py \
  --model "$MODEL" \
  --data "$DATA" \
  --train-start "2021-07-01" \
  --train-end "2021-10-01" \
  --eval-start "2023-07-01" \
  --eval-end "2023-10-01" \
  --output "$OUTPUT"

echo ""
echo "完了！"
echo "出力: $OUTPUT"

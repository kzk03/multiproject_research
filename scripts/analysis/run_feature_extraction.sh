#!/bin/bash
# 50proj no_osモデルから特徴量を抽出

set -e

echo "========================================"
echo "IRLモデルから14次元特徴量を抽出"
echo "========================================"
echo ""

MODEL="outputs/50projects_irl/no_os/train_0-3m/irl_model.pt"
DATA="data/openstack_50proj_2021_2024_feat.csv"
OUTPUT="outputs/analysis_data/developer_state_features_0-3m.csv"

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
  --train-start "2021-01-01" \
  --train-end "2021-04-01" \
  --eval-start "2023-01-01" \
  --eval-end "2023-04-01" \
  --output "$OUTPUT"

echo ""
echo "完了！"
echo "出力: $OUTPUT"

#!/bin/bash
# 訓練データと評価データの特徴量を別々に抽出（データリークなし）

set -e

MODEL="outputs/50projects_irl_timeseries/2x_os/train_6-9m/irl_model.pt"
DATA="data/openstack_50proj_2021_2024_feat.csv"

echo "=========================================="
echo "訓練データ特徴量抽出"
echo "=========================================="

uv run python scripts/analysis/extract_state_features.py \
  --model "$MODEL" \
  --data "$DATA" \
  --train-start "2021-07-01" \
  --train-end "2021-10-01" \
  --eval-start "2021-07-01" \
  --eval-end "2021-10-01" \
  --output "outputs/analysis_data/train_features_6-9m.csv"

echo "訓練データ特徴量: $(wc -l < outputs/analysis_data/train_features_6-9m.csv) 行"

echo ""
echo "=========================================="
echo "評価データ特徴量抽出"
echo "=========================================="

uv run python scripts/analysis/extract_state_features.py \
  --model "$MODEL" \
  --data "$DATA" \
  --train-start "2021-07-01" \
  --train-end "2021-10-01" \
  --eval-start "2023-07-01" \
  --eval-end "2023-10-01" \
  --output "outputs/analysis_data/eval_features_6-9m.csv"

echo "評価データ特徴量: $(wc -l < outputs/analysis_data/eval_features_6-9m.csv) 行"

echo ""
echo "=========================================="
echo "Random Forest 正しい比較実行"
echo "=========================================="

uv run python scripts/analysis/compare_irl_vs_rf_correct.py \
  --train-features "outputs/analysis_data/train_features_6-9m.csv" \
  --eval-features "outputs/analysis_data/eval_features_6-9m.csv" \
  --output "outputs/analysis_data/rf_correct_comparison"

echo "完了！"

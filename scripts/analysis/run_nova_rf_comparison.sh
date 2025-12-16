#!/bin/bash
#
# Nova単体プロジェクト: IRL vs Random Forest 比較
#
# 実行手順:
# 1. Nova単体データから訓練・評価期間の特徴量を抽出
# 2. Random Forestで訓練・評価
# 3. IRLの結果と比較
#

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# ディレクトリ設定
DATA_DIR="data/thesis"
OUTPUT_DIR="outputs/analysis_data/nova_single_rf_comparison"
RESULTS_DIR="results/review_continuation_cross_eval_nova"

# Nova単体データパス
NOVA_DATA="${DATA_DIR}/review_requests_openstack_multi_5y_detail.csv"

# IRL結果パス（6-9m → 6-9mパターン）
IRL_RESULTS="${RESULTS_DIR}/train_6-9m/eval_6-9m/metrics.json"

# 出力パス
TRAIN_FEATURES="${OUTPUT_DIR}/train_features_6-9m_nova.csv"
EVAL_FEATURES="${OUTPUT_DIR}/eval_features_6-9m_nova.csv"
RF_OUTPUT="${OUTPUT_DIR}/rf_results"

# 出力ディレクトリ作成
mkdir -p "$OUTPUT_DIR"
mkdir -p "$RF_OUTPUT"

echo "========================================"
echo "Nova単体: IRL vs Random Forest 比較"
echo "========================================"
echo ""
echo "データ: $NOVA_DATA"
echo "出力: $OUTPUT_DIR"
echo ""

# ステップ1: IRLモデルパス確認
echo "[1/4] IRLモデルパス確認..."
IRL_MODEL="${RESULTS_DIR}/train_6-9m/irl_model.pt"
if [ ! -f "$IRL_MODEL" ]; then
    echo "ERROR: IRLモデルが見つかりません: $IRL_MODEL"
    exit 1
fi
echo "  モデル: $IRL_MODEL"
echo ""

# ステップ2: 訓練データ特徴量抽出（2020年7月～10月）
echo "[2/4] 訓練データ特徴量抽出（2020-07-01 ~ 2020-10-01）..."
uv run python scripts/analysis/extract_state_features_nova_single.py \
  --model "$IRL_MODEL" \
  --data "$NOVA_DATA" \
  --train-start "2020-07-01" \
  --train-end "2020-10-01" \
  --eval-start "2020-07-01" \
  --eval-end "2020-10-01" \
  --output "$TRAIN_FEATURES" \
  --project-filter "openstack/nova"

echo ""
echo "訓練データ抽出完了: $TRAIN_FEATURES"
echo ""

# サンプル数確認
echo "訓練データサンプル数:"
uv run python -c "
import pandas as pd
df = pd.read_csv('$TRAIN_FEATURES')
print(f'  Total: {len(df)}')
print(f'  Positive: {df[\"true_label\"].sum()} ({df[\"true_label\"].mean()*100:.1f}%)')
print(f'  Negative: {(~df[\"true_label\"].astype(bool)).sum()} ({(1-df[\"true_label\"].mean())*100:.1f}%)')
"
echo ""

# ステップ3: 評価データ特徴量抽出（2022年7月～10月）
echo "[3/4] 評価データ特徴量抽出（2022-07-01 ~ 2022-10-01）..."
uv run python scripts/analysis/extract_state_features_nova_single.py \
  --model "$IRL_MODEL" \
  --data "$NOVA_DATA" \
  --train-start "2020-07-01" \
  --train-end "2020-10-01" \
  --eval-start "2022-07-01" \
  --eval-end "2022-10-01" \
  --output "$EVAL_FEATURES" \
  --project-filter "openstack/nova"

echo ""
echo "評価データ抽出完了: $EVAL_FEATURES"
echo ""

# サンプル数確認
echo "評価データサンプル数:"
uv run python -c "
import pandas as pd
df = pd.read_csv('$EVAL_FEATURES')
print(f'  Total: {len(df)}')
print(f'  Positive: {df[\"true_label\"].sum()} ({df[\"true_label\"].mean()*100:.1f}%)')
print(f'  Negative: {(~df[\"true_label\"].astype(bool)).sum()} ({(1-df[\"true_label\"].mean())*100:.1f}%)')
"
echo ""

# ステップ4: Random Forest訓練・評価
echo "[4/4] Random Forest訓練・評価（状態10次元 + 行動4次元）..."
uv run python scripts/analysis/compare_irl_vs_rf_nova_single.py \
  --train-features "$TRAIN_FEATURES" \
  --eval-features "$EVAL_FEATURES" \
  --output "$RF_OUTPUT"

echo ""
echo "========================================"
echo "完了！"
echo "========================================"
echo ""
echo "結果ファイル:"
echo "  - RF結果: ${RF_OUTPUT}/rf_nova_single_results.json"
echo "  - 特徴量重要度: ${RF_OUTPUT}/rf_nova_single_feature_importance.csv"
echo ""

# IRL結果と比較
if [ -f "$IRL_RESULTS" ]; then
    echo "IRL結果 (6-9m → 6-9m):"
    cat "$IRL_RESULTS" | python -m json.tool | grep -E '"(f1_score|auc_roc|precision|recall)"'
    echo ""
fi

echo "RF結果:"
cat "${RF_OUTPUT}/rf_nova_single_results.json" | python -m json.tool | grep -E '"(f1|auc_roc|precision|recall)"'
echo ""
echo "詳細は以下を参照:"
echo "  - $OUTPUT_DIR"

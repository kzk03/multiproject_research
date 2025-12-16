#!/bin/bash
#
# 50プロジェクト IRL学習パイプライン
# データ収集完了後に実行
#

set -e

echo "=============================================="
echo "50プロジェクト IRL学習パイプライン"
echo "=============================================="
echo ""

# 設定
DATA_RAW="data/openstack_50proj_2021_2024.csv"
DATA_FEAT="data/openstack_50proj_2021_2024_feat.csv"
OUTPUT_DIR="outputs/50projects_irl"

# データ存在確認
if [ ! -f "${DATA_RAW}" ]; then
    echo "エラー: データファイルが見つかりません: ${DATA_RAW}"
    echo "先に ./scripts/collect_50projects_data.sh を実行してください"
    exit 1
fi

echo "入力データ: ${DATA_RAW}"
echo "出力先: ${OUTPUT_DIR}"
echo ""

# ==============================================
# Phase 1: マルチプロジェクト特徴量追加
# ==============================================
echo "=============================================="
echo "Phase 1: マルチプロジェクト特徴量追加"
echo "=============================================="

if [ -f "${DATA_FEAT}" ]; then
    echo "既存の特徴量データが見つかりました: ${DATA_FEAT}"
    echo "スキップしますか？ (y/n)"
    read -r SKIP_FEATURES
    if [ "$SKIP_FEATURES" != "y" ]; then
        echo "特徴量を再計算します..."
        mv "${DATA_FEAT}" "${DATA_FEAT}.backup.$(date +%Y%m%d_%H%M%S)"
        uv run python scripts/pipeline/add_multiproject_features.py \
            --input "${DATA_RAW}" \
            --output "${DATA_FEAT}"
    fi
else
    echo "マルチプロジェクト特徴量を追加中..."
    uv run python scripts/pipeline/add_multiproject_features.py \
        --input "${DATA_RAW}" \
        --output "${DATA_FEAT}"
fi

echo "特徴量追加完了: ${DATA_FEAT}"
echo ""

# 特徴量確認
.venv/bin/python - <<EOF
import pandas as pd
df = pd.read_csv('${DATA_FEAT}')
print(f"特徴量次元数: {len(df.columns)}次元")
print(f"レビュー数: {len(df):,}件")
print(f"期間: {df['request_time'].min()} ～ {df['request_time'].max()}")
EOF

echo ""

# ==============================================
# Phase 2: IRL学習（Cross-Temporal）
# ==============================================
echo "=============================================="
echo "Phase 2: IRL学習（Cross-Temporal）"
echo "=============================================="
echo "訓練: 2021年"
echo "評価: 2023年"
echo ""

mkdir -p "${OUTPUT_DIR}"

# No Oversampling
echo "--- [1/3] No Oversampling ---"
uv run python scripts/train/train_cross_temporal_multiproject.py \
    --reviews "${DATA_FEAT}" \
    --train-base-start 2021-01-01 \
    --eval-base-start 2023-01-01 \
    --total-months 12 \
    --output "${OUTPUT_DIR}/no_os" \
    --negative-oversample-factor 1

echo ""

# 2x Oversampling
echo "--- [2/3] 2x Oversampling ---"
uv run python scripts/train/train_cross_temporal_multiproject.py \
    --reviews "${DATA_FEAT}" \
    --train-base-start 2021-01-01 \
    --eval-base-start 2023-01-01 \
    --total-months 12 \
    --output "${OUTPUT_DIR}/2x_os" \
    --negative-oversample-factor 2

echo ""

# 3x Oversampling
echo "--- [3/3] 3x Oversampling ---"
uv run python scripts/train/train_cross_temporal_multiproject.py \
    --reviews "${DATA_FEAT}" \
    --train-base-start 2021-01-01 \
    --eval-base-start 2023-01-01 \
    --total-months 12 \
    --output "${OUTPUT_DIR}/3x_os" \
    --negative-oversample-factor 3

echo ""

# ==============================================
# Phase 3: 結果サマリー
# ==============================================
echo "=============================================="
echo "結果サマリー"
echo "=============================================="

.venv/bin/python - <<EOF
import json
from pathlib import Path

output_dir = Path('${OUTPUT_DIR}')
experiments = ['no_os', '2x_os', '3x_os']

print()
print('50プロジェクト IRL 学習結果')
print('=' * 80)
print(f"{'実験':<15} {'F1':>8} {'Precision':>10} {'Recall':>8} {'AUC-ROC':>8}")
print('-' * 80)

for exp in experiments:
    metrics_file = output_dir / exp / 'metrics.json'
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
        f1 = metrics.get('f1_score', 0.0)
        precision = metrics.get('precision', 0.0)
        recall = metrics.get('recall', 0.0)
        auc_roc = metrics.get('roc_auc', 0.0)
        print(f"{exp:<15} {f1:>8.3f} {precision:>10.3f} {recall:>8.3f} {auc_roc:>8.3f}")
    else:
        print(f"{exp:<15} (結果なし)")

print('=' * 80)
print()

# 20プロジェクトとの比較
print('参考: 20プロジェクト結果（Multi-Project）')
print('-' * 80)
baseline_dir = Path('outputs/multiproject_irl_full')
if baseline_dir.exists():
    for exp in experiments:
        metrics_file = baseline_dir / exp / 'metrics.json'
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
            f1 = metrics.get('f1_score', 0.0)
            precision = metrics.get('precision', 0.0)
            recall = metrics.get('recall', 0.0)
            auc_roc = metrics.get('roc_auc', 0.0)
            print(f"{exp:<15} {f1:>8.3f} {precision:>10.3f} {recall:>8.3f} {auc_roc:>8.3f}")
    print()

EOF

echo ""
echo "=============================================="
echo "完了！"
echo "=============================================="
echo "結果保存先: ${OUTPUT_DIR}"
echo ""
echo "詳細結果:"
echo "  - ${OUTPUT_DIR}/no_os/metrics.json"
echo "  - ${OUTPUT_DIR}/2x_os/metrics.json"
echo "  - ${OUTPUT_DIR}/3x_os/metrics.json"
echo ""

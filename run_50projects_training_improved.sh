#!/bin/bash
#
# 50プロジェクト IRL学習パイプライン（AUC-ROC改善版）
# パラメータ調整のみで性能改善を図る
#

set -e

echo "=============================================="
echo "50プロジェクト IRL学習（AUC-ROC改善版）"
echo "=============================================="
echo ""

# 設定
DATA_RAW="data/openstack_50proj_2021_2024.csv"
DATA_FEAT="data/openstack_50proj_2021_2024_feat.csv"
OUTPUT_DIR="outputs/50projects_irl_improved"

echo "入力データ: ${DATA_FEAT}"
echo "出力先: ${OUTPUT_DIR}"
echo ""

# 特徴量データの存在確認
if [ ! -f "${DATA_FEAT}" ]; then
    echo "エラー: 特徴量データが見つかりません: ${DATA_FEAT}"
    exit 1
fi

# ==============================================
# AUC-ROC改善のためのパラメータ調整
# ==============================================
echo "=============================================="
echo "AUC-ROC改善のためのパラメータ設定"
echo "=============================================="
echo ""
echo "【変更点】"
echo "1. エポック数: 20 → 50（より十分な学習）"
echo "2. 学習率: 0.0001 → 0.0003（より細かい調整）"
echo "3. Focal Loss alpha: auto → 0.3（負例により焦点）"
echo "4. Focal Loss gamma: auto → 2.0（難しい例により焦点）"
echo ""
echo "【期待効果】"
echo "- エポック増加: モデルが十分に収束"
echo "- 学習率調整: ランキング性能の細かい最適化"
echo "- Focal alpha減少: 正例・負例のスコア差を拡大"
echo "- Focal gamma増加: 難しい例（境界付近）を重点学習"
echo ""

mkdir -p "${OUTPUT_DIR}"

# ==============================================
# IRL学習（3パターン × 改善版パラメータ）
# ==============================================
echo "=============================================="
echo "IRL学習開始"
echo "=============================================="
echo "訓練: 2021年"
echo "評価: 2023年"
echo ""

# No Oversampling（改善版）
echo "--- [1/3] No Oversampling (Improved) ---"
uv run python scripts/train/train_cross_temporal_multiproject.py \
    --reviews "${DATA_FEAT}" \
    --train-base-start 2021-01-01 \
    --eval-base-start 2023-01-01 \
    --total-months 12 \
    --output "${OUTPUT_DIR}/no_os" \
    --negative-oversample-factor 1 \
    --epochs 50 \
    --learning-rate 0.0003 \
    --focal-alpha 0.3 \
    --focal-gamma 2.0

echo ""

# 2x Oversampling（改善版）
echo "--- [2/3] 2x Oversampling (Improved) ---"
uv run python scripts/train/train_cross_temporal_multiproject.py \
    --reviews "${DATA_FEAT}" \
    --train-base-start 2021-01-01 \
    --eval-base-start 2023-01-01 \
    --total-months 12 \
    --output "${OUTPUT_DIR}/2x_os" \
    --negative-oversample-factor 2 \
    --epochs 50 \
    --learning-rate 0.0003 \
    --focal-alpha 0.3 \
    --focal-gamma 2.0

echo ""

# 3x Oversampling（改善版）
echo "--- [3/3] 3x Oversampling (Improved) ---"
uv run python scripts/train/train_cross_temporal_multiproject.py \
    --reviews "${DATA_FEAT}" \
    --train-base-start 2021-01-01 \
    --eval-base-start 2023-01-01 \
    --total-months 12 \
    --output "${OUTPUT_DIR}/3x_os" \
    --negative-oversample-factor 3 \
    --epochs 50 \
    --learning-rate 0.0003 \
    --focal-alpha 0.3 \
    --focal-gamma 2.0

echo ""

# ==============================================
# 結果サマリー
# ==============================================
echo "=============================================="
echo "結果サマリー"
echo "=============================================="

.venv/bin/python - <<PYEOF
import json
from pathlib import Path

output_dir = Path('${OUTPUT_DIR}')
experiments = ['no_os', '2x_os', '3x_os']

print()
print('50プロジェクト IRL 学習結果（改善版）')
print('=' * 80)
print(f"{'実験':<15} {'F1':>8} {'Precision':>10} {'Recall':>8} {'AUC-ROC':>8}")
print('-' * 80)

for exp in experiments:
    # 対角線要素の平均を計算
    f1_sum, prec_sum, rec_sum, auc_sum = 0, 0, 0, 0
    count = 0
    
    for period in ['0-3m', '3-6m', '6-9m', '9-12m']:
        metrics_file = output_dir / exp / f'train_{period}' / f'eval_{period}' / 'metrics.json'
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
            f1_sum += metrics.get('f1_score', 0.0)
            prec_sum += metrics.get('precision', 0.0)
            rec_sum += metrics.get('recall', 0.0)
            auc_sum += metrics.get('auc_roc', 0.0)
            count += 1
    
    if count > 0:
        f1_avg = f1_sum / count
        prec_avg = prec_sum / count
        rec_avg = rec_sum / count
        auc_avg = auc_sum / count
        print(f"{exp:<15} {f1_avg:>8.3f} {prec_avg:>10.3f} {rec_avg:>8.3f} {auc_avg:>8.3f}")
    else:
        print(f"{exp:<15} (結果なし)")

print('=' * 80)
print()

# 元のモデルとの比較
print('参考: 元のモデル（50プロジェクト）')
print('-' * 80)
baseline_dir = Path('outputs/50projects_irl')
if baseline_dir.exists():
    for exp in experiments:
        f1_sum, prec_sum, rec_sum, auc_sum = 0, 0, 0, 0
        count = 0
        
        for period in ['0-3m', '3-6m', '6-9m', '9-12m']:
            metrics_file = baseline_dir / exp / f'train_{period}' / f'eval_{period}' / 'metrics.json'
            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics = json.load(f)
                f1_sum += metrics.get('f1_score', 0.0)
                prec_sum += metrics.get('precision', 0.0)
                rec_sum += metrics.get('recall', 0.0)
                auc_sum += metrics.get('auc_roc', 0.0)
                count += 1
        
        if count > 0:
            f1_avg = f1_sum / count
            prec_avg = prec_sum / count
            rec_avg = rec_sum / count
            auc_avg = auc_sum / count
            print(f"{exp:<15} {f1_avg:>8.3f} {prec_avg:>10.3f} {rec_avg:>8.3f} {auc_avg:>8.3f}")
    print()

PYEOF

echo ""
echo "=============================================="
echo "完了！"
echo "=============================================="
echo "結果保存先: ${OUTPUT_DIR}"
echo ""
echo "詳細結果:"
echo "  - ${OUTPUT_DIR}/no_os/matrix_*.csv"
echo "  - ${OUTPUT_DIR}/2x_os/matrix_*.csv"
echo "  - ${OUTPUT_DIR}/3x_os/matrix_*.csv"
echo ""
echo "パラメータ設定:"
echo "  - エポック数: 50"
echo "  - 学習率: 0.0003"
echo "  - Focal Loss: alpha=0.3, gamma=2.0"
echo ""

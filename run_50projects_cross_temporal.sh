#!/bin/bash
# 50プロジェクト クロス時間評価の実行スクリプト

set -e  # エラーで停止

# カラー出力
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 設定
REVIEWS_CSV="${1:-data/openstack_50proj_2021_2024_feat.csv}"
TRAIN_BASE_START="${2:-2021-01-01}"
EVAL_BASE_START="${3:-2023-01-01}"
TOTAL_MONTHS="${4:-12}"
OUTPUT_DIR="${5:-outputs/50projects_cross_temporal}"
PROJECT="${6:-}"  # 空の場合は全プロジェクト
EPOCHS="${7:-20}"

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}50プロジェクト クロス時間評価パイプライン${NC}"
echo -e "${GREEN}============================================================${NC}"
echo "レビューCSV: $REVIEWS_CSV"
echo "訓練期間ベース: $TRAIN_BASE_START"
echo "評価期間ベース: $EVAL_BASE_START"
echo "総期間: $TOTAL_MONTHS ヶ月"
echo "出力ディレクトリ: $OUTPUT_DIR"
echo "プロジェクト: ${PROJECT:-全50プロジェクト}"
echo "エポック数: $EPOCHS"
echo -e "${GREEN}============================================================${NC}"

# データファイルの存在確認
if [ ! -f "$REVIEWS_CSV" ]; then
    echo -e "${RED}エラー: レビューCSVファイルが見つかりません: $REVIEWS_CSV${NC}"
    echo -e "${YELLOW}データセットを構築してください:${NC}"
    echo "  ./scripts/collect_50projects_data.sh"
    echo "  uv run python scripts/pipeline/add_multiproject_features.py \\"
    echo "    --input data/openstack_50proj_2021_2024.csv \\"
    echo "    --output data/openstack_50proj_2021_2024_feat.csv"
    exit 1
fi

# データ統計表示
echo -e "\n${GREEN}データセット統計:${NC}"
python3 - <<EOF
import pandas as pd
df = pd.read_csv('$REVIEWS_CSV')
print(f"  総レビュー数: {len(df):,}件")
print(f"  プロジェクト数: {df['project'].nunique()}プロジェクト")
print(f"  レビュアー数: {df['reviewer_email'].nunique():,}名")
print(f"  期間: {df['request_time'].min()} ～ {df['request_time'].max()}")
print(f"  特徴量次元数: {len(df.columns)}次元")
EOF

# ステップ1: クロス時間評価の実行
echo -e "\n${GREEN}[1/2] クロス時間評価を実行中...${NC}"
echo "  訓練期間: $TRAIN_BASE_START から $TOTAL_MONTHS ヶ月"
echo "  評価期間: $EVAL_BASE_START から $TOTAL_MONTHS ヶ月"
echo ""

if [ -z "$PROJECT" ]; then
    # 全プロジェクト
    uv run python scripts/train/train_cross_temporal_multiproject.py \
        --reviews "$REVIEWS_CSV" \
        --train-base-start "$TRAIN_BASE_START" \
        --eval-base-start "$EVAL_BASE_START" \
        --total-months "$TOTAL_MONTHS" \
        --output "$OUTPUT_DIR" \
        --epochs "$EPOCHS"
else
    # 特定プロジェクト
    uv run python scripts/train/train_cross_temporal_multiproject.py \
        --reviews "$REVIEWS_CSV" \
        --train-base-start "$TRAIN_BASE_START" \
        --eval-base-start "$EVAL_BASE_START" \
        --total-months "$TOTAL_MONTHS" \
        --output "$OUTPUT_DIR" \
        --project "$PROJECT" \
        --epochs "$EPOCHS"
fi

# ステップ2: ヒートマップの作成
echo -e "\n${GREEN}[2/2] ヒートマップを作成中...${NC}"

uv run python scripts/analysis/create_cross_temporal_heatmaps.py \
    --input "$OUTPUT_DIR"

# 完了メッセージ
echo -e "\n${GREEN}============================================================${NC}"
echo -e "${GREEN}50プロジェクト クロス時間評価が完了しました！${NC}"
echo -e "${GREEN}============================================================${NC}"
echo -e "結果: ${YELLOW}$OUTPUT_DIR${NC}"
echo ""
echo "生成されたファイル:"
echo "  - matrix_*.csv           (メトリクスマトリクス)"
echo "  - summary_statistics.json (サマリー統計)"
echo "  - heatmaps/*.png         (ヒートマップ)"
echo ""
echo "クロス時間評価結果サマリー:"
python3 - <<EOF
import json
from pathlib import Path

stats_file = Path('$OUTPUT_DIR') / 'summary_statistics.json'
if stats_file.exists():
    with open(stats_file) as f:
        stats = json.load(f)

    print("\n主要メトリクス（平均値）:")
    for metric in ['AUC_ROC', 'F1_Score', 'Precision', 'Recall']:
        if metric in stats:
            mean_val = stats[metric].get('mean', 0.0)
            std_val = stats[metric].get('std', 0.0)
            min_val = stats[metric].get('min', 0.0)
            max_val = stats[metric].get('max', 0.0)
            print(f"  {metric:12s}: {mean_val:.3f} (±{std_val:.3f}) [min: {min_val:.3f}, max: {max_val:.3f}]")

    print("\nベストパフォーマンス:")
    if 'best_combinations' in stats:
        for metric, info in stats['best_combinations'].items():
            train = info.get('train_period', 'N/A')
            eval_p = info.get('eval_period', 'N/A')
            value = info.get('value', 0.0)
            print(f"  {metric:12s}: {value:.3f} (訓練: {train}, 評価: {eval_p})")
else:
    print("  統計ファイルが見つかりません")
EOF

echo ""
echo "次のステップ:"
echo "  1. ヒートマップを確認: open $OUTPUT_DIR/heatmaps/heatmap_4_metrics.png"
echo "  2. AUC-ROCマトリクスを確認: cat $OUTPUT_DIR/matrix_AUC_ROC.csv"
echo "  3. F1マトリクスを確認: cat $OUTPUT_DIR/matrix_F1_Score.csv"
echo "  4. 統計を確認: cat $OUTPUT_DIR/summary_statistics.json"
echo ""
echo "20プロジェクトとの比較:"
echo "  既存の20プロジェクト結果: outputs/multiproject_irl_full/"
echo "  50プロジェクト結果: $OUTPUT_DIR"
echo -e "${GREEN}============================================================${NC}"

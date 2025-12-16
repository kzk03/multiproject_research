#!/bin/bash
# クロス時間評価の実行スクリプト

set -e  # エラーで停止

# カラー出力
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 設定
REVIEWS_CSV="${1:-data/review_requests_openstack_multi_5y_detail.csv}"
TRAIN_BASE_START="${2:-2021-01-01}"
EVAL_BASE_START="${3:-2023-01-01}"
TOTAL_MONTHS="${4:-12}"
OUTPUT_DIR="${5:-results/cross_temporal_multiproject}"
PROJECT="${6:-}"  # 空の場合は全プロジェクト
EPOCHS="${7:-20}"

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}クロス時間評価パイプライン${NC}"
echo -e "${GREEN}============================================================${NC}"
echo "レビューCSV: $REVIEWS_CSV"
echo "訓練期間ベース: $TRAIN_BASE_START"
echo "評価期間ベース: $EVAL_BASE_START"
echo "総期間: $TOTAL_MONTHS ヶ月"
echo "出力ディレクトリ: $OUTPUT_DIR"
echo "プロジェクト: ${PROJECT:-全プロジェクト}"
echo "エポック数: $EPOCHS"
echo -e "${GREEN}============================================================${NC}"

# データファイルの存在確認
if [ ! -f "$REVIEWS_CSV" ]; then
    echo -e "${RED}エラー: レビューCSVファイルが見つかりません: $REVIEWS_CSV${NC}"
    echo -e "${YELLOW}データセットを構築してください:${NC}"
    echo "  uv run python scripts/pipeline/build_dataset.py \\"
    echo "    --gerrit-url https://review.opendev.org \\"
    echo "    --project openstack/nova openstack/neutron openstack/cinder \\"
    echo "    --start-date 2021-01-01 \\"
    echo "    --end-date 2024-01-01 \\"
    echo "    --output $REVIEWS_CSV"
    exit 1
fi

# ステップ1: クロス時間評価の実行
echo -e "\n${GREEN}[1/2] クロス時間評価を実行中...${NC}"

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
echo -e "${GREEN}クロス時間評価が完了しました！${NC}"
echo -e "${GREEN}============================================================${NC}"
echo -e "結果: ${YELLOW}$OUTPUT_DIR${NC}"
echo ""
echo "生成されたファイル:"
echo "  - matrix_*.csv           (メトリクスマトリクス)"
echo "  - summary_statistics.json (サマリー統計)"
echo "  - heatmaps/*.png         (ヒートマップ)"
echo ""
echo "次のステップ:"
echo "  1. ヒートマップを確認: open $OUTPUT_DIR/heatmaps/heatmap_4_metrics.png"
echo "  2. マトリクスを確認: cat $OUTPUT_DIR/matrix_AUC_ROC.csv"
echo "  3. 統計を確認: cat $OUTPUT_DIR/summary_statistics.json"
echo -e "${GREEN}============================================================${NC}"

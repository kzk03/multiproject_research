#!/bin/bash
#
# Android 50プロジェクトのデータ収集
#
# 期間設定:
#   全体: 2021-01-01 ～ 2024-01-01（36ヶ月）
#   訓練: 2021-01-01 ～ 2023-01-01（24ヶ月）
#   評価: 2023-01-01 ～ 2024-01-01（12ヶ月）
#

set -e

echo "=========================================="
echo "Android 50プロジェクトデータ収集"
echo "=========================================="
echo ""

# 設定
GERRIT_URL="https://android-review.googlesource.com"
START_DATE="2021-01-01"
END_DATE="2024-01-01"
OUTPUT_FILE="data/android_50proj_2021_2024.csv"
PROJECT_LIST="projects_android_50.txt"

# プロジェクトリストをファイルから読み込み
if [ ! -f "${PROJECT_LIST}" ]; then
    echo "エラー: ${PROJECT_LIST} が見つかりません"
    exit 1
fi

PROJECTS=()
while IFS= read -r line; do
    PROJECTS+=("$line")
done < "${PROJECT_LIST}"

echo "対象プロジェクト数: ${#PROJECTS[@]}"
echo ""
echo "期間: ${START_DATE} ～ ${END_DATE}"
echo "出力: ${OUTPUT_FILE}"
echo ""
echo "プロジェクトリスト（先頭10件）:"
for i in "${!PROJECTS[@]}"; do
    if [ $i -lt 10 ]; then
        echo "  ${PROJECTS[$i]}"
    fi
done
echo ""

# データ収集実行
echo "データ収集を開始..."
echo ""

uv run python scripts/pipeline/build_dataset.py \
    --gerrit-url "${GERRIT_URL}" \
    --project "${PROJECTS[@]}" \
    --start-date "${START_DATE}" \
    --end-date "${END_DATE}" \
    --output "${OUTPUT_FILE}" \
    --response-window 14

echo ""
echo "=========================================="
echo "収集完了！"
echo "=========================================="
echo ""

# データサマリーを表示
if command -v python3 &> /dev/null; then
    echo "データサマリー:"
    python3 - <<EOF
import pandas as pd
df = pd.read_csv('${OUTPUT_FILE}')
print(f"総レコード数: {len(df):,}")
print(f"開発者数: {df['reviewer_email'].nunique():,}")
print(f"プロジェクト数: {df['project'].nunique()}")
print(f"正例数: {(df['label']==1).sum():,} ({(df['label']==1).mean():.1%})")
print(f"負例数: {(df['label']==0).sum():,} ({(df['label']==0).mean():.1%})")
print(f"期間: {df['request_time'].min()} ～ {df['request_time'].max()}")
print()
print("プロジェクト別レビュー数（Top 10）:")
top10 = df['project'].value_counts().head(10)
for proj, count in top10.items():
    proj_df = df[df['project'] == proj]
    print(f"  {proj}: {count:,}件, 正例率 {(proj_df['label']==1).mean():.1%}")
EOF
fi

echo ""
echo "次のステップ:"
echo "1. データ確認: head -n 20 ${OUTPUT_FILE}"
echo "2. マルチプロジェクト特徴量追加: uv run python scripts/pipeline/add_multiproject_features.py --input ${OUTPUT_FILE} --output data/android_50proj_2021_2024_feat.csv"
echo "3. IRL学習: uv run python scripts/train/train_cross_temporal_multiproject.py --train-csv data/android_50proj_2021_2024_feat.csv --train-year 2021 --test-year 2023 --output-dir outputs/android_50projects_irl"
echo ""

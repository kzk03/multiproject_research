#!/bin/bash
#
# 論文と同じ期間設定で複数プロジェクトのデータ収集
#
# 期間設定（論文と同じ）:
#   全体: 2021-01-01 ～ 2024-01-01（36ヶ月）
#   訓練: 2021-01-01 ～ 2023-01-01（24ヶ月）
#   評価: 2023-01-01 ～ 2024-01-01（12ヶ月）
#
# プロジェクト個別判定 + 複数プロジェクト横断学習:
#   - 各プロジェクトで継続（承諾）を個別判定
#   - プロジェクト間の相互作用も見るため、全プロジェクトを1つのCSVに統合
#   - is_cross_project フラグでプロジェクト横断活動を識別
#

set -e

echo "=========================================="
echo "複数プロジェクトデータ収集（論文期間）"
echo "=========================================="
echo ""

# 設定
GERRIT_URL="https://review.opendev.org"
START_DATE="2021-01-01"
END_DATE="2024-01-01"
OUTPUT_FILE="data/multiproject_paper_data.csv"

# 対象プロジェクト（OpenStack主要5プロジェクト）
PROJECTS=(
    "openstack/nova"
    "openstack/neutron"
    "openstack/cinder"
    "openstack/keystone"
    "openstack/glance"
)

echo "対象プロジェクト:"
for project in "${PROJECTS[@]}"; do
    echo "  - $project"
done
echo ""

echo "期間: ${START_DATE} ～ ${END_DATE}"
echo "出力: ${OUTPUT_FILE}"
echo ""

# データ収集実行（既存のbuild_dataset.pyを使用）
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
print("プロジェクト別:")
for proj in df['project'].unique():
    proj_df = df[df['project'] == proj]
    print(f"  {proj}: {len(proj_df):,}件, 正例率 {(proj_df['label']==1).mean():.1%}")
print()
if 'is_cross_project' in df.columns:
    print(f"クロスプロジェクト活動: {df['is_cross_project'].sum():,}件 ({df['is_cross_project'].mean():.1%})")
EOF
fi

echo ""
echo "次のステップ:"
echo "1. データ確認: head -n 20 ${OUTPUT_FILE}"
echo "2. IRL形式変換: uv run python scripts/convert_to_irl_format.py --input ${OUTPUT_FILE} --output data/multiproject_irl_data.json"
echo "3. モデル訓練: 既存の訓練スクリプトを使用"
echo ""

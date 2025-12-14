#!/bin/bash
#
# 論文と同じ期間設定で50プロジェクトのデータ収集
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
echo "50プロジェクトデータ収集（論文期間）"
echo "=========================================="
echo ""

# 設定
GERRIT_URL="https://review.opendev.org"
START_DATE="2021-01-01"
END_DATE="2024-01-01"
OUTPUT_FILE="data/openstack_50proj_2021_2024.csv"

# 対象プロジェクト（OpenStack公式50プロジェクト）
PROJECTS=(
    # Tier 1: Interop必須コア（25）
    "openstack/nova"
    "openstack/python-novaclient"
    "openstack/placement"
    "openstack/nova-specs"
    "openstack/neutron"
    "openstack/python-neutronclient"
    "openstack/neutron-lib"
    "openstack/neutron-specs"
    "openstack/cinder"
    "openstack/python-cinderclient"
    "openstack/os-brick"
    "openstack/cinder-specs"
    "openstack/swift"
    "openstack/python-swiftclient"
    "openstack/swift-specs"
    "openstack/keystone"
    "openstack/python-keystoneclient"
    "openstack/keystoneauth"
    "openstack/glance"
    "openstack/python-glanceclient"
    "openstack/glance_store"
    "openstack/glance-specs"
    "openstack/horizon"
    "openstack/django_openstack_auth"
    "openstack/horizon-specs"

    # Tier 2: 重要サービス（15）
    "openstack/heat"
    "openstack/python-heatclient"
    "openstack/heat-specs"
    "openstack/ironic"
    "openstack/python-ironicclient"
    "openstack/ironic-inspector"
    "openstack/octavia"
    "openstack/python-octaviaclient"
    "openstack/manila"
    "openstack/python-manilaclient"
    "openstack/barbican"
    "openstack/python-barbicanclient"
    "openstack/designate"
    "openstack/python-designateclient"
    "openstack/magnum"

    # Tier 3: インフラ・SDK（7）
    "openstack/openstacksdk"
    "openstack/python-openstackclient"
    "openstack/oslo.config"
    "openstack/oslo.messaging"
    "openstack/oslo.db"
    "openstack/requirements"
    "openstack/releases"

    # Tier 4: デプロイメント（3）
    "openstack/kolla"
    "openstack/kolla-ansible"
    "openstack/devstack"
)

echo "対象プロジェクト数: ${#PROJECTS[@]}"
echo ""
echo "Tier 1 (Interop必須コア): 25プロジェクト"
echo "Tier 2 (重要サービス): 15プロジェクト"
echo "Tier 3 (インフラ・SDK): 7プロジェクト"
echo "Tier 4 (デプロイメント): 3プロジェクト"
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
print("Tier別レビュー数:")
tier1_projects = ['openstack/nova', 'openstack/neutron', 'openstack/cinder', 'openstack/swift', 'openstack/keystone', 'openstack/glance', 'openstack/horizon']
tier1_df = df[df['project'].str.split('/').str[-1].str.split('-').str[0].isin([p.split('/')[-1].split('-')[0] for p in tier1_projects])]
print(f"  Tier 1 (Core): {len(tier1_df):,}件")

print()
print("プロジェクト別レビュー数（Top 10）:")
top10 = df['project'].value_counts().head(10)
for proj, count in top10.items():
    proj_df = df[df['project'] == proj]
    print(f"  {proj}: {count:,}件, 正例率 {(proj_df['label']==1).mean():.1%}")
print()
if 'is_cross_project' in df.columns:
    print(f"クロスプロジェクト活動: {df['is_cross_project'].sum():,}件 ({df['is_cross_project'].mean():.1%})")
EOF
fi

echo ""
echo "次のステップ:"
echo "1. データ確認: head -n 20 ${OUTPUT_FILE}"
echo "2. マルチプロジェクト特徴量追加: uv run python scripts/pipeline/add_multiproject_features.py --input ${OUTPUT_FILE} --output data/openstack_50proj_2021_2024_feat.csv"
echo "3. IRL学習: uv run python scripts/train/train_cross_temporal_multiproject.py --train-csv data/openstack_50proj_2021_2024_feat.csv --train-year 2021 --test-year 2023 --output-dir outputs/50projects_irl"
echo ""

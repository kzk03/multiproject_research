#!/usr/bin/env python3
"""
特徴量抽出のテストスクリプト

データソースから特徴量が正しく抽出できるか確認します。
- 欠損値のチェック
- 特徴量の分布確認
- 実際のデータでの動作確認
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.review_predictor.features.common_features import (
    extract_common_features,
    FEATURE_NAMES,
)

# データ読み込み
data_path = Path("/Users/kazuki-h/research/multiproject_research/data/review_requests_openstack_multi_5y_detail.csv")
df = pd.read_csv(data_path)

# Nova単体にフィルタ
df = df[df['project'] == 'openstack/nova'].copy()

# タイムスタンプ列を統一
df['timestamp'] = pd.to_datetime(df['request_time'])
df['email'] = df['reviewer_email']

print("=" * 80)
print("データソース確認")
print("=" * 80)
print(f"総レコード数: {len(df)}")
print(f"期間: {df['timestamp'].min()} ～ {df['timestamp'].max()}")
print(f"開発者数: {df['email'].nunique()}")
print(f"\n利用可能な列:")
for col in df.columns:
    print(f"  - {col}")

# テスト: 1人の開発者の特徴量を抽出
print("\n" + "=" * 80)
print("特徴量抽出テスト")
print("=" * 80)

cutoff_date = pd.Timestamp("2023-01-01")
feature_start = cutoff_date - pd.DateOffset(months=12)
feature_end = cutoff_date

# 活動が多い開発者を選択
dev_counts = df[(df['timestamp'] >= feature_start) & (df['timestamp'] < feature_end)]['email'].value_counts()
if len(dev_counts) == 0:
    print("ERROR: 指定期間に開発者が見つかりません")
    sys.exit(1)

test_email = dev_counts.index[0]
print(f"\nテスト対象開発者: {test_email}")
print(f"活動回数: {dev_counts.iloc[0]}")

# 特徴量抽出
try:
    features = extract_common_features(
        df=df,
        email=test_email,
        feature_start=feature_start,
        feature_end=feature_end,
        normalize=False
    )

    print("\n✓ 特徴量抽出成功！")
    print("\n抽出された特徴量:")
    for feature_name in FEATURE_NAMES:
        value = features.get(feature_name, 'N/A')
        print(f"  {feature_name:30s} = {value}")

    # 欠損値チェック
    missing_features = [k for k, v in features.items() if pd.isna(v) or v is None]
    if missing_features:
        print(f"\n⚠️  欠損特徴量: {missing_features}")
    else:
        print("\n✓ 欠損なし！")

except Exception as e:
    print(f"\n✗ 特徴量抽出エラー: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 複数開発者でテスト
print("\n" + "=" * 80)
print("複数開発者での特徴量抽出テスト（上位10名）")
print("=" * 80)

features_list = []
for i, email in enumerate(dev_counts.index[:10]):
    try:
        features = extract_common_features(
            df=df,
            email=email,
            feature_start=feature_start,
            feature_end=feature_end,
            normalize=False
        )
        features['email'] = email
        features_list.append(features)
        print(f"  {i+1}. {email}: ✓")
    except Exception as e:
        print(f"  {i+1}. {email}: ✗ ({e})")

features_df = pd.DataFrame(features_list)

# 統計サマリー
print("\n" + "=" * 80)
print("特徴量統計サマリー")
print("=" * 80)
print(features_df[FEATURE_NAMES].describe())

# 欠損率チェック
print("\n" + "=" * 80)
print("欠損率チェック")
print("=" * 80)
missing_rates = features_df[FEATURE_NAMES].isna().mean() * 100
for feature_name, rate in missing_rates.items():
    status = "⚠️" if rate > 0 else "✓"
    print(f"  {status} {feature_name:30s}: {rate:5.1f}%")

print("\n" + "=" * 80)
print("テスト完了！")
print("=" * 80)

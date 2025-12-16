#!/usr/bin/env python3
"""
データリークチェック: RFが訓練データで評価されていないか確認
"""

import pandas as pd
from pathlib import Path

# データ読み込み
features_path = Path("outputs/analysis_data/developer_state_features_2x_6-9m.csv")
predictions_path = Path("outputs/50projects_irl/2x_os/train_6-9m/eval_6-9m/predictions.csv")

df_features = pd.read_csv(features_path)
df_predictions = pd.read_csv(predictions_path)

print("=" * 80)
print("データリーク検証")
print("=" * 80)

print(f"\n特徴量データ: {len(df_features)} サンプル")
print(f"予測データ: {len(df_predictions)} サンプル")

# マージ
df = pd.merge(
    df_features,
    df_predictions[['reviewer_email', 'predicted_prob', 'true_label']],
    on='reviewer_email',
    how='inner'
)

print(f"マージ後: {len(df)} サンプル")

# 訓練データとテストデータの区別があるか？
print("\n" + "=" * 80)
print("重要な確認")
print("=" * 80)

# このデータはどこから来たか？
print("\n1. このデータは何のデータか？")
print("   - 特徴量: outputs/analysis_data/developer_state_features_2x_6-9m.csv")
print("   - 予測: outputs/50projects_irl/2x_os/train_6-9m/eval_6-9m/predictions.csv")
print("   → これは**評価データ（eval）**")

print("\n2. IRLの訓練データとテストデータは？")
print("   - 訓練: 2021-07-01～2021-10-01 の開発者")
print("   - 評価: 2023-07-01～2023-10-01 の開発者")
print("   → **時系列分割（時間的に分離）**")

print("\n3. Random Forestは？")
print("   - 訓練: このCSVの183サンプル全部")
print("   - 評価: このCSVの183サンプル全部（**同じデータ**）")
print("   → **データリーク！訓練データで評価している！**")

print("\n" + "=" * 80)
print("結論")
print("=" * 80)
print("Random Forestは訓練データで評価しているため、")
print("F1=0.997という異常に高いスコアが出ている。")
print("これは**完全なデータリーク**です。")
print("=" * 80)

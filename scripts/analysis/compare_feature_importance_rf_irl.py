#!/usr/bin/env python3
"""
RFとIRLの特徴量重要度比較分析

RF: Gini係数ベースの重要度（正の値、合計=1.0）
IRL: 勾配ベースの重要度（正負の値、絶対値で比較）
"""

import json
import pandas as pd
from pathlib import Path

# データ読み込み
rf_data = pd.read_csv('/Users/kazuki-h/research/multiproject_research/outputs/analysis_data/nova_single_rf_comparison/rf_results/rf_nova_single_feature_importance.csv')

with open('/Users/kazuki-h/research/multiproject_research/results/review_continuation_cross_eval_nova/average_feature_importance/gradient_importance_average.json', 'r') as f:
    irl_data = json.load(f)

# RF特徴量重要度（既にソート済み）
print("=" * 80)
print("Random Forest 特徴量重要度（Gini係数ベース）")
print("=" * 80)
print("\n上位10特徴量:")
for i, row in rf_data.head(10).iterrows():
    print(f"  {i+1:2d}. {row['feature']:35s}: {row['importance']:.6f} ({row['importance']*100:.2f}%)")

# IRL特徴量重要度（絶対値でソート）
print("\n" + "=" * 80)
print("IRL (LSTM) 特徴量重要度（勾配ベース、絶対値）")
print("=" * 80)

# 状態と行動を統合
irl_combined = {}
irl_combined.update(irl_data['state_importance'])
irl_combined.update(irl_data['action_importance'])

# 絶対値でソート
irl_sorted = sorted(irl_combined.items(), key=lambda x: abs(x[1]), reverse=True)

print("\n上位10特徴量（絶対値）:")
for i, (feature, importance) in enumerate(irl_sorted[:10], 1):
    sign = "+" if importance > 0 else "-"
    print(f"  {i:2d}. {feature:35s}: {sign}{abs(importance):.6f} (絶対値: {abs(importance):.6f})")

# 特徴量マッピング（RFとIRLの対応）
feature_mapping = {
    'avg_response_time': '応答速度',
    'total_reviews': '総レビュー数',
    'recent_activity_frequency': '最近の活動頻度',
    'experience_days': '経験日数',
    'avg_activity_gap': '平均活動間隔',
    'recent_acceptance_rate': '最近の受諾率',
    'avg_action_intensity': '強度（ファイル数）',
    'review_load': 'レビュー負荷',
    'avg_review_size': 'レビュー規模（行数）',
    'activity_trend': '活動トレンド',
    'total_changes': '総レビュー依頼数',
    'collaboration_score': '協力スコア',
    'code_quality_score': '総承諾率',
    'avg_collaboration': '協力度',
}

# 比較表作成
print("\n" + "=" * 80)
print("RF vs IRL 特徴量重要度比較（共通特徴量のみ）")
print("=" * 80)

comparison_data = []
for _, row in rf_data.iterrows():
    rf_feature = row['feature']
    if rf_feature in feature_mapping:
        irl_feature = feature_mapping[rf_feature]
        if irl_feature in irl_combined:
            comparison_data.append({
                'feature': irl_feature,
                'rf_importance': row['importance'],
                'rf_rank': _ + 1,
                'irl_importance_raw': irl_combined[irl_feature],
                'irl_importance_abs': abs(irl_combined[irl_feature]),
            })

# IRL絶対値でソート
comparison_df = pd.DataFrame(comparison_data)
comparison_df['irl_rank'] = comparison_df['irl_importance_abs'].rank(ascending=False, method='min').astype(int)
comparison_df = comparison_df.sort_values('rf_rank')

print("\n特徴量比較表:")
print(f"{'順位(RF)':8s} {'特徴量':30s} {'RF重要度':12s} {'順位(IRL)':10s} {'IRL重要度':15s} {'差':10s}")
print("-" * 95)

for _, row in comparison_df.iterrows():
    rf_rank = row['rf_rank']
    irl_rank = row['irl_rank']
    rank_diff = irl_rank - rf_rank
    rank_diff_str = f"{rank_diff:+d}" if rank_diff != 0 else "="

    irl_sign = "+" if row['irl_importance_raw'] > 0 else "-"

    print(f"{rf_rank:3d}位     {row['feature']:30s} {row['rf_importance']:7.4f} ({row['rf_importance']*100:5.2f}%)  "
          f"{irl_rank:3d}位      {irl_sign}{abs(row['irl_importance_raw']):7.6f}      {rank_diff_str:>8s}")

# トップ3の違いを強調
print("\n" + "=" * 80)
print("【重要な発見】順位の逆転")
print("=" * 80)

print("\nRF Top 5:")
top5_rf = comparison_df.nsmallest(5, 'rf_rank')
for _, row in top5_rf.iterrows():
    print(f"  {int(row['rf_rank']):2d}. {row['feature']:30s}: {row['rf_importance']*100:5.2f}%  (IRL順位: {int(row['irl_rank']):2d}位)")

print("\nIRL Top 5 (絶対値):")
top5_irl = comparison_df.nsmallest(5, 'irl_rank')
for _, row in top5_irl.iterrows():
    irl_sign = "+" if row['irl_importance_raw'] > 0 else "-"
    print(f"  {int(row['irl_rank']):2d}. {row['feature']:30s}: {irl_sign}{abs(row['irl_importance_raw']):.6f}  (RF順位: {int(row['rf_rank']):2d}位)")

# CSV保存
output_dir = Path('/Users/kazuki-h/research/multiproject_research/outputs/analysis_data/feature_importance_comparison')
output_dir.mkdir(parents=True, exist_ok=True)

comparison_df.to_csv(output_dir / 'rf_vs_irl_feature_importance.csv', index=False)
print(f"\n比較結果を保存: {output_dir / 'rf_vs_irl_feature_importance.csv'}")

# 統計サマリー
print("\n" + "=" * 80)
print("統計サマリー")
print("=" * 80)

print(f"\nRF:")
print(f"  上位5特徴量の合計重要度: {comparison_df.nsmallest(5, 'rf_rank')['rf_importance'].sum()*100:.2f}%")
print(f"  上位10特徴量の合計重要度: {comparison_df.nsmallest(10, 'rf_rank')['rf_importance'].sum()*100:.2f}%")

print(f"\nIRL (絶対値):")
print(f"  上位5特徴量の合計重要度: {comparison_df.nsmallest(5, 'irl_rank')['irl_importance_abs'].sum():.6f}")
print(f"  上位10特徴量の合計重要度: {comparison_df.nsmallest(10, 'irl_rank')['irl_importance_abs'].sum():.6f}")

print("\n" + "=" * 80)
print("分析完了")
print("=" * 80)

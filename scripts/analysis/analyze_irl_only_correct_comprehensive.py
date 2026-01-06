"""
IRLだけが正しく予測できた開発者の包括的な分析スクリプト
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 日本語フォント設定
plt.rcParams['font.family'] = 'Hiragino Sans'
plt.rcParams['axes.unicode_minus'] = False

# データ読み込み
base_dir = Path("/Users/kazuki-h/research/multiproject_research/outputs/irl_rf_10pattern_analysis")
irl_only = pd.read_csv(base_dir / "irl_only_correct.csv")
both_correct = pd.read_csv(base_dir / "both_correct.csv")
rf_only = pd.read_csv(base_dir / "rf_only_correct.csv")

print("=" * 80)
print("IRL Only Correct: 基本統計")
print("=" * 80)
print(f"総件数: {len(irl_only)}")
print(f"ユニーク開発者数: {irl_only['reviewer_id'].nunique()}")
print(f"\n真のラベル分布:")
print(irl_only['true_label'].value_counts())
print(f"\n継続率: {(irl_only['true_label'] == 1).sum() / len(irl_only) * 100:.1f}%")

# 1. 時系列パターン別分析
print("\n" + "=" * 80)
print("時系列パターン別分布")
print("=" * 80)
pattern_dist = irl_only['pattern'].value_counts().sort_index()
for pattern, count in pattern_dist.items():
    pct = count / len(irl_only) * 100
    print(f"{pattern}: {count}件 ({pct:.1f}%)")

# 2. IRLとRFの確率分布比較
print("\n" + "=" * 80)
print("IRL vs RF 確率分布統計")
print("=" * 80)
print("\nIRL確率:")
print(irl_only['irl_prob'].describe())
print("\nRF確率:")
print(irl_only['rf_prob'].describe())

# 3. 継続/離脱別の詳細分析
print("\n" + "=" * 80)
print("継続開発者（true_label=1）の分析")
print("=" * 80)
continuers = irl_only[irl_only['true_label'] == 1]
print(f"件数: {len(continuers)}")
print(f"\n平均IRL確率: {continuers['irl_prob'].mean():.4f}")
print(f"平均RF確率: {continuers['rf_prob'].mean():.4f}")
print(f"平均レビュー数: {continuers['history_count'].mean():.1f}")
print(f"平均継続率: {continuers['history_rate'].mean():.4f}")

print("\nRF確率が低い上位5件（RFの誤り度が大きい）:")
top_continuers = continuers.nsmallest(5, 'rf_prob')[['reviewer_id', 'irl_prob', 'rf_prob', 'history_count', 'history_rate', 'pattern']]
print(top_continuers.to_string(index=False))

print("\n" + "=" * 80)
print("離脱開発者（true_label=0）の分析")
print("=" * 80)
leavers = irl_only[irl_only['true_label'] == 0]
print(f"件数: {len(leavers)}")
print(f"\n平均IRL確率: {leavers['irl_prob'].mean():.4f}")
print(f"平均RF確率: {leavers['rf_prob'].mean():.4f}")
print(f"平均レビュー数: {leavers['history_count'].mean():.1f}")
print(f"平均継続率: {leavers['history_rate'].mean():.4f}")

print("\nRF確率が高い上位5件（RFの誤り度が大きい）:")
top_leavers = leavers.nlargest(5, 'rf_prob')[['reviewer_id', 'irl_prob', 'rf_prob', 'history_count', 'history_rate', 'pattern']]
print(top_leavers.to_string(index=False))

# 4. レビュー数による分類
print("\n" + "=" * 80)
print("レビュー数別分類")
print("=" * 80)

def categorize_review_count(count):
    if count < 10:
        return "Very Low (< 10)"
    elif count < 50:
        return "Low (10-49)"
    elif count < 100:
        return "Medium (50-99)"
    elif count < 500:
        return "High (100-499)"
    else:
        return "Very High (≥ 500)"

irl_only['review_category'] = irl_only['history_count'].apply(categorize_review_count)
review_stats = irl_only.groupby('review_category').agg({
    'reviewer_id': 'count',
    'irl_prob': 'mean',
    'rf_prob': 'mean',
    'history_count': 'mean',
    'history_rate': 'mean',
    'true_label': 'mean'
}).round(4)
review_stats.columns = ['件数', '平均IRL確率', '平均RF確率', '平均レビュー数', '平均継続率', '継続率']
print(review_stats)

# 5. CI/Bot系開発者の識別
print("\n" + "=" * 80)
print("CI/Bot系開発者の分析")
print("=" * 80)

ci_keywords = ['ci', 'bot', 'test', 'jenkins', 'zuul']
irl_only['is_ci'] = irl_only['reviewer_id'].str.lower().apply(
    lambda x: any(keyword in x for keyword in ci_keywords)
)

ci_devs = irl_only[irl_only['is_ci']]
human_devs = irl_only[~irl_only['is_ci']]

print(f"CI/Bot系: {len(ci_devs)}件")
print(f"人間開発者: {len(human_devs)}件")

print("\nCI/Bot系の統計:")
ci_stats = ci_devs.agg({
    'irl_prob': 'mean',
    'rf_prob': 'mean',
    'history_count': 'mean',
    'history_rate': 'mean',
    'true_label': 'mean'
}).round(4)
print(ci_stats)

print("\n人間開発者の統計:")
human_stats = human_devs.agg({
    'irl_prob': 'mean',
    'rf_prob': 'mean',
    'history_count': 'mean',
    'history_rate': 'mean',
    'true_label': 'mean'
}).round(4)
print(human_stats)

# 6. 同一開発者の時系列追跡
print("\n" + "=" * 80)
print("複数期間に出現する開発者")
print("=" * 80)

multi_period_devs = irl_only.groupby('reviewer_id').size()
multi_period_devs = multi_period_devs[multi_period_devs > 1].sort_values(ascending=False)

print(f"複数期間に出現: {len(multi_period_devs)}名")
for dev_id, count in multi_period_devs.items():
    dev_data = irl_only[irl_only['reviewer_id'] == dev_id].sort_values('pattern')
    print(f"\n{dev_id} ({count}回出現):")
    for _, row in dev_data.iterrows():
        print(f"  {row['pattern']}: true={row['true_label']}, IRL={row['irl_prob']:.4f}, RF={row['rf_prob']:.4f}")

# 7. 可視化
output_dir = Path("/Users/kazuki-h/research/multiproject_research/results/nova_single_rf_vs_irl_comparison")
output_dir.mkdir(parents=True, exist_ok=True)

# 図1: IRL vs RF 確率分布（継続/離脱別）
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, label, title in zip(axes, [1, 0], ['継続開発者 (true_label=1)', '離脱開発者 (true_label=0)']):
    subset = irl_only[irl_only['true_label'] == label]

    ax.scatter(subset['irl_prob'], subset['rf_prob'], alpha=0.6, s=100)
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='閾値 0.5')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)

    # 正解領域を色分け
    if label == 1:  # 継続
        ax.axhspan(0, 0.5, alpha=0.1, color='red', label='RF誤り領域')
        ax.axvspan(0.5, 1, alpha=0.1, color='green', label='IRL正解領域')
    else:  # 離脱
        ax.axhspan(0.5, 1, alpha=0.1, color='red', label='RF誤り領域')
        ax.axvspan(0, 0.5, alpha=0.1, color='green', label='IRL正解領域')

    ax.set_xlabel('IRL確率', fontsize=12)
    ax.set_ylabel('RF確率', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "irl_only_correct_scatter.png", dpi=300, bbox_inches='tight')
print(f"\n図1を保存: {output_dir / 'irl_only_correct_scatter.png'}")

# 図2: レビュー数 vs 継続率（継続/離脱別）
fig, ax = plt.subplots(figsize=(12, 6))

continuers = irl_only[irl_only['true_label'] == 1]
leavers = irl_only[irl_only['true_label'] == 0]

ax.scatter(continuers['history_count'], continuers['history_rate'],
           alpha=0.7, s=100, c='blue', label='継続 (IRL正解)', marker='o')
ax.scatter(leavers['history_count'], leavers['history_rate'],
           alpha=0.7, s=100, c='red', label='離脱 (IRL正解)', marker='s')

ax.set_xlabel('レビュー数 (history_count)', fontsize=12)
ax.set_ylabel('継続率 (history_rate)', fontsize=12)
ax.set_title('IRLだけが正解: レビュー数 vs 継続率', fontsize=14, fontweight='bold')
ax.set_xscale('log')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "irl_only_correct_review_vs_rate.png", dpi=300, bbox_inches='tight')
print(f"図2を保存: {output_dir / 'irl_only_correct_review_vs_rate.png'}")

# 図3: RF確率の分布（継続/離脱別）
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(continuers['rf_prob'], bins=20, alpha=0.7, color='blue', edgecolor='black')
axes[0].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='閾値 0.5')
axes[0].set_xlabel('RF確率', fontsize=12)
axes[0].set_ylabel('頻度', fontsize=12)
axes[0].set_title('継続開発者: RFの誤った低確率分布', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].hist(leavers['rf_prob'], bins=20, alpha=0.7, color='red', edgecolor='black')
axes[1].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='閾値 0.5')
axes[1].set_xlabel('RF確率', fontsize=12)
axes[1].set_ylabel('頻度', fontsize=12)
axes[1].set_title('離脱開発者: RFの誤った高確率分布', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "irl_only_correct_rf_prob_dist.png", dpi=300, bbox_inches='tight')
print(f"図3を保存: {output_dir / 'irl_only_correct_rf_prob_dist.png'}")

# 図4: レビュー数カテゴリ別の確率比較
fig, ax = plt.subplots(figsize=(12, 6))

categories = ['Very Low (< 10)', 'Low (10-49)', 'Medium (50-99)', 'High (100-499)', 'Very High (≥ 500)']
review_stats_sorted = review_stats.reindex(categories)

x = np.arange(len(categories))
width = 0.35

ax.bar(x - width/2, review_stats_sorted['平均IRL確率'], width, label='IRL確率', alpha=0.8, color='blue')
ax.bar(x + width/2, review_stats_sorted['平均RF確率'], width, label='RF確率', alpha=0.8, color='orange')
ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='閾値 0.5')

ax.set_xlabel('レビュー数カテゴリ', fontsize=12)
ax.set_ylabel('平均確率', fontsize=12)
ax.set_title('レビュー数カテゴリ別: IRL vs RF確率', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=15, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / "irl_only_correct_category_comparison.png", dpi=300, bbox_inches='tight')
print(f"図4を保存: {output_dir / 'irl_only_correct_category_comparison.png'}")

# 図5: 確率差の分析
irl_only['prob_diff'] = irl_only['rf_prob'] - irl_only['irl_prob']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 継続開発者の確率差（RFが低すぎる）
continuers_diff = continuers['rf_prob'] - continuers['irl_prob']
axes[0].hist(continuers_diff, bins=15, alpha=0.7, color='blue', edgecolor='black')
axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='差=0')
axes[0].set_xlabel('確率差 (RF - IRL)', fontsize=12)
axes[0].set_ylabel('頻度', fontsize=12)
axes[0].set_title('継続開発者: RFがIRLより低く予測（負の差）', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 離脱開発者の確率差（RFが高すぎる）
leavers_diff = leavers['rf_prob'] - leavers['irl_prob']
axes[1].hist(leavers_diff, bins=15, alpha=0.7, color='red', edgecolor='black')
axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='差=0')
axes[1].set_xlabel('確率差 (RF - IRL)', fontsize=12)
axes[1].set_ylabel('頻度', fontsize=12)
axes[1].set_title('離脱開発者: RFがIRLより高く予測（正の差）', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "irl_only_correct_prob_difference.png", dpi=300, bbox_inches='tight')
print(f"図5を保存: {output_dir / 'irl_only_correct_prob_difference.png'}")

print("\n" + "=" * 80)
print("分析完了！")
print("=" * 80)
print(f"出力ディレクトリ: {output_dir}")
print(f"- IRL_ONLY_CORRECT_DETAILED_ANALYSIS.md")
print(f"- irl_only_correct_scatter.png")
print(f"- irl_only_correct_review_vs_rate.png")
print(f"- irl_only_correct_rf_prob_dist.png")
print(f"- irl_only_correct_category_comparison.png")
print(f"- irl_only_correct_prob_difference.png")

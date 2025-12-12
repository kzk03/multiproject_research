"""
過去承諾率のパラドックス分析

なぜ過去の承諾率が高いレビュアーが、評価期間では低い承諾率になるのか？
"""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

# スタイル設定
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_single_project_predictions():
    """Single Project (Nova)の予測データを読み込む"""
    base_path = Path("/Users/kazuki-h/research/multiproject_research/results/review_continuation_cross_eval_nova")

    train_periods = ['0-3m', '3-6m', '6-9m', '9-12m']
    eval_periods = ['0-3m', '3-6m', '6-9m', '9-12m']

    all_predictions = []

    for train in train_periods:
        for eval_p in eval_periods:
            pred_file = base_path / f"train_{train}" / f"eval_{eval_p}" / "predictions.csv"
            if pred_file.exists():
                df = pd.read_csv(pred_file)
                df['train_period'] = train
                df['eval_period'] = eval_p
                all_predictions.append(df)

    return pd.concat(all_predictions, ignore_index=True)

def categorize_past_acceptance(df: pd.DataFrame) -> pd.DataFrame:
    """過去の承諾率でカテゴリ化"""
    df = df.copy()
    df['past_acceptance_category'] = pd.cut(
        df['history_acceptance_rate'],
        bins=[-0.01, 0.3, 0.5, 0.7, 1.01],
        labels=['Low (<30%)', 'Medium (30-50%)', 'High (50-70%)', 'Very High (>70%)']
    )

    # 評価期間での実際の承諾率を計算
    df['eval_acceptance_rate'] = df['eval_accepted_count'] / df['eval_request_count']

    return df

def analyze_acceptance_rate_change(df: pd.DataFrame, output_dir: Path):
    """承諾率の変化を分析"""
    print("\n" + "="*100)
    print("過去承諾率のパラドックス分析")
    print("="*100)

    df_cat = categorize_past_acceptance(df)

    # カテゴリごとの分析
    print("\n【カテゴリごとの統計】")
    print("-"*100)

    results = []

    for category in df_cat['past_acceptance_category'].cat.categories:
        subset = df_cat[df_cat['past_acceptance_category'] == category]

        if len(subset) == 0:
            continue

        print(f"\n{category}:")
        print(f"  サンプル数: {len(subset)}")
        print(f"  過去承諾率（平均）: {subset['history_acceptance_rate'].mean():.3f}")
        print(f"  過去承諾率（範囲）: {subset['history_acceptance_rate'].min():.3f} - {subset['history_acceptance_rate'].max():.3f}")
        print(f"  評価期間承諾率（平均）: {subset['eval_acceptance_rate'].mean():.3f}")
        print(f"  評価期間承諾率（中央値）: {subset['eval_acceptance_rate'].median():.3f}")
        print(f"  評価期間承諾率（範囲）: {subset['eval_acceptance_rate'].min():.3f} - {subset['eval_acceptance_rate'].max():.3f}")
        print(f"  実際にAcceptedしたケース: {subset['true_label'].sum()} / {len(subset)} ({subset['true_label'].mean():.1%})")

        # レビュー数の統計
        print(f"  過去レビュー数（平均）: {subset['history_request_count'].mean():.1f}")
        print(f"  評価期間レビュー数（平均）: {subset['eval_request_count'].mean():.1f}")

        # 承諾率の変化
        acceptance_change = subset['eval_acceptance_rate'].mean() - subset['history_acceptance_rate'].mean()
        print(f"  承諾率の変化: {acceptance_change:+.3f}")

        results.append({
            'category': category,
            'count': len(subset),
            'past_acceptance_mean': subset['history_acceptance_rate'].mean(),
            'eval_acceptance_mean': subset['eval_acceptance_rate'].mean(),
            'eval_acceptance_median': subset['eval_acceptance_rate'].median(),
            'actual_accepted_rate': subset['true_label'].mean(),
            'past_review_count': subset['history_request_count'].mean(),
            'eval_review_count': subset['eval_request_count'].mean(),
            'acceptance_change': acceptance_change
        })

    return pd.DataFrame(results), df_cat

def plot_acceptance_rate_transition(df_cat: pd.DataFrame, output_dir: Path):
    """承諾率の遷移を可視化"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Acceptance Rate Paradox Analysis', fontsize=16, fontweight='bold')

    # 1. カテゴリごとの承諾率変化
    ax1 = axes[0, 0]

    categories = df_cat['past_acceptance_category'].cat.categories
    past_rates = []
    eval_rates = []

    for cat in categories:
        subset = df_cat[df_cat['past_acceptance_category'] == cat]
        if len(subset) > 0:
            past_rates.append(subset['history_acceptance_rate'].mean())
            eval_rates.append(subset['eval_acceptance_rate'].mean())

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax1.bar(x - width/2, past_rates, width, label='Past (History)', alpha=0.8, color='#4ECDC4')
    bars2 = ax1.bar(x + width/2, eval_rates, width, label='Current (Eval)', alpha=0.8, color='#FF6B6B')

    ax1.set_xlabel('Past Acceptance Rate Category', fontweight='bold')
    ax1.set_ylabel('Acceptance Rate', fontweight='bold')
    ax1.set_title('Acceptance Rate: Past vs Current', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.replace(' ', '\n') for c in categories], fontsize=9)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # 値をバーに表示
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9)

    # 2. 散布図: 過去 vs 評価期間の承諾率
    ax2 = axes[0, 1]

    colors = {'Low (<30%)': '#FF6B6B', 'Medium (30-50%)': '#4ECDC4',
              'High (50-70%)': '#45B7D1', 'Very High (>70%)': '#96CEB4'}

    for cat in categories:
        subset = df_cat[df_cat['past_acceptance_category'] == cat]
        if len(subset) > 0:
            ax2.scatter(subset['history_acceptance_rate'],
                       subset['eval_acceptance_rate'],
                       label=cat, alpha=0.6, s=50, color=colors.get(cat, 'gray'))

    # 対角線（変化なしのライン）
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=2, label='No Change')

    ax2.set_xlabel('Past Acceptance Rate (History)', fontweight='bold')
    ax2.set_ylabel('Current Acceptance Rate (Eval)', fontweight='bold')
    ax2.set_title('Individual Reviewer Transitions', fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-0.05, 1.05])
    ax2.set_ylim([-0.05, 1.05])

    # 3. レビュー数の変化
    ax3 = axes[1, 0]

    past_review_counts = []
    eval_review_counts = []

    for cat in categories:
        subset = df_cat[df_cat['past_acceptance_category'] == cat]
        if len(subset) > 0:
            past_review_counts.append(subset['history_request_count'].mean())
            eval_review_counts.append(subset['eval_request_count'].mean())

    bars1 = ax3.bar(x - width/2, past_review_counts, width, label='Past Period', alpha=0.8, color='#4ECDC4')
    bars2 = ax3.bar(x + width/2, eval_review_counts, width, label='Eval Period', alpha=0.8, color='#FF6B6B')

    ax3.set_xlabel('Past Acceptance Rate Category', fontweight='bold')
    ax3.set_ylabel('Average Review Count', fontweight='bold')
    ax3.set_title('Review Count: Past vs Current', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([c.replace(' ', '\n') for c in categories], fontsize=9)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=9)

    # 4. ヒストグラム: High/Very Highカテゴリの評価期間承諾率
    ax4 = axes[1, 1]

    high_subset = df_cat[df_cat['past_acceptance_category'].isin(['High (50-70%)', 'Very High (>70%)'])]

    ax4.hist(high_subset['eval_acceptance_rate'], bins=20, alpha=0.7, color='#FF6B6B', edgecolor='black')
    ax4.axvline(high_subset['eval_acceptance_rate'].mean(), color='blue', linestyle='--',
               linewidth=2, label=f'Mean: {high_subset["eval_acceptance_rate"].mean():.3f}')
    ax4.axvline(high_subset['eval_acceptance_rate'].median(), color='green', linestyle='--',
               linewidth=2, label=f'Median: {high_subset["eval_acceptance_rate"].median():.3f}')

    ax4.set_xlabel('Current Acceptance Rate (Eval Period)', fontweight='bold')
    ax4.set_ylabel('Frequency', fontweight='bold')
    ax4.set_title('Distribution: High/Very High Past → Current', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'acceptance_rate_paradox.png', dpi=300, bbox_inches='tight')
    print(f"\n保存: {output_dir / 'acceptance_rate_paradox.png'}")

def analyze_time_dependency(df_cat: pd.DataFrame, output_dir: Path):
    """時系列での変化を分析"""
    print("\n" + "="*100)
    print("時系列での承諾率変化")
    print("="*100)

    # High/Very Highカテゴリに注目
    high_subset = df_cat[df_cat['past_acceptance_category'].isin(['High (50-70%)', 'Very High (>70%)'])]

    print(f"\nHigh/Very Highカテゴリのレビュアー: {len(high_subset)}名")

    # 訓練期間と評価期間の関係
    period_analysis = high_subset.groupby(['train_period', 'eval_period']).agg({
        'eval_acceptance_rate': ['mean', 'median', 'count'],
        'true_label': 'mean'
    }).round(3)

    print("\n訓練期間×評価期間の承諾率:")
    print(period_analysis)

    # 時系列プロット
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Temporal Analysis: High/Very High Past Acceptance Reviewers', fontsize=16, fontweight='bold')

    train_periods = ['0-3m', '3-6m', '6-9m', '9-12m']

    for idx, train in enumerate(train_periods):
        ax = axes[idx // 2, idx % 2]

        subset = high_subset[high_subset['train_period'] == train]

        if len(subset) == 0:
            continue

        eval_period_stats = subset.groupby('eval_period').agg({
            'eval_acceptance_rate': 'mean',
            'true_label': 'mean'
        })

        x = range(len(eval_period_stats))

        ax.plot(x, eval_period_stats['eval_acceptance_rate'], marker='o',
               label='Eval Acceptance Rate', linewidth=2, markersize=8)
        ax.plot(x, eval_period_stats['true_label'], marker='s',
               label='Actual Acceptance (true_label)', linewidth=2, markersize=8)

        ax.set_xlabel('Evaluation Period', fontweight='bold')
        ax.set_ylabel('Acceptance Rate', fontweight='bold')
        ax.set_title(f'Train Period: {train}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(eval_period_stats.index)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_dir / 'temporal_acceptance_paradox.png', dpi=300, bbox_inches='tight')
    print(f"保存: {output_dir / 'temporal_acceptance_paradox.png'}")

def analyze_individual_reviewers(df_cat: pd.DataFrame, output_dir: Path):
    """個別レビュアーの詳細分析"""
    print("\n" + "="*100)
    print("個別レビュアーの行動変化分析")
    print("="*100)

    # High/Very Highで評価期間の承諾率が低いレビュアーを抽出
    high_subset = df_cat[df_cat['past_acceptance_category'].isin(['High (50-70%)', 'Very High (>70%)'])]
    low_current = high_subset[high_subset['eval_acceptance_rate'] < 0.3]

    print(f"\n過去High/Very High → 現在Low (<30%)のレビュアー: {len(low_current)}名")

    if len(low_current) > 0:
        print("\nTop 10ケース（最も大きな低下）:")
        print("-"*100)

        low_current_sorted = low_current.copy()
        low_current_sorted['acceptance_drop'] = low_current_sorted['history_acceptance_rate'] - low_current_sorted['eval_acceptance_rate']
        low_current_sorted = low_current_sorted.sort_values('acceptance_drop', ascending=False).head(10)

        for idx, row in low_current_sorted.iterrows():
            print(f"\nレビュアー: {row['reviewer_email']}")
            print(f"  過去承諾率: {row['history_acceptance_rate']:.3f}")
            print(f"  現在承諾率: {row['eval_acceptance_rate']:.3f}")
            print(f"  低下幅: {row['acceptance_drop']:.3f}")
            print(f"  過去レビュー数: {row['history_request_count']}")
            print(f"  評価期間レビュー数: {row['eval_request_count']}")
            print(f"  評価期間 承諾/拒否: {row['eval_accepted_count']}/{row['eval_rejected_count']}")
            print(f"  訓練期間: {row['train_period']} → 評価期間: {row['eval_period']}")

    # 逆のケース: Low → High
    low_past = df_cat[df_cat['past_acceptance_category'] == 'Low (<30%)']
    high_current = low_past[low_past['eval_acceptance_rate'] > 0.7]

    print(f"\n\n過去Low (<30%) → 現在High (>70%)のレビュアー: {len(high_current)}名")

    if len(high_current) > 0:
        print("\nTop 5ケース（最も大きな向上）:")
        print("-"*100)

        high_current_sorted = high_current.copy()
        high_current_sorted['acceptance_rise'] = high_current_sorted['eval_acceptance_rate'] - high_current_sorted['history_acceptance_rate']
        high_current_sorted = high_current_sorted.sort_values('acceptance_rise', ascending=False).head(5)

        for idx, row in high_current_sorted.iterrows():
            print(f"\nレビュアー: {row['reviewer_email']}")
            print(f"  過去承諾率: {row['history_acceptance_rate']:.3f}")
            print(f"  現在承諾率: {row['eval_acceptance_rate']:.3f}")
            print(f"  向上幅: {row['acceptance_rise']:.3f}")
            print(f"  過去レビュー数: {row['history_request_count']}")
            print(f"  評価期間レビュー数: {row['eval_request_count']}")

def main():
    output_dir = Path("/Users/kazuki-h/research/multiproject_research/docs/figures/paradox_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("データを読み込み中...")
    df = load_single_project_predictions()
    print(f"総予測数: {len(df)}")

    # 分析実行
    results_df, df_cat = analyze_acceptance_rate_change(df, output_dir)

    # 可視化
    print("\n可視化を生成中...")
    plot_acceptance_rate_transition(df_cat, output_dir)

    # 時系列分析
    analyze_time_dependency(df_cat, output_dir)

    # 個別レビュアー分析
    analyze_individual_reviewers(df_cat, output_dir)

    # 結果をCSVに保存
    results_df.to_csv(output_dir / 'acceptance_rate_paradox_summary.csv', index=False)
    print(f"\n保存: {output_dir / 'acceptance_rate_paradox_summary.csv'}")

    print(f"\n全ての分析が完了しました: {output_dir}")

if __name__ == "__main__":
    main()

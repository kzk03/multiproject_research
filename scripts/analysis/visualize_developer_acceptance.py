"""
開発者ごとのレビュー承諾率を視覚化するスクリプト
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_visualizations():
    data_dir = Path("/Users/kazuki-h/research/multiproject_research/results/developer_acceptance_rate")
    output_dir = data_dir / "visualizations"
    output_dir.mkdir(exist_ok=True)

    # データを読み込み
    all_data = pd.read_csv(data_dir / "all_developers_acceptance.csv")
    summary = pd.read_csv(data_dir / "project_acceptance_summary.csv")

    # 色設定
    colors = {'Qt': '#FF6B6B', 'Android': '#4ECDC4', 'Chromium': '#FFA07A', 'OpenStack': '#45B7D1'}

    # 図1: プロジェクト別承諾率の分布
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1-1: ヒストグラム
    ax = axes[0, 0]
    for project in ['Qt', 'Android', 'Chromium', 'OpenStack']:
        project_data = all_data[all_data['project'] == project]['acceptance_rate'] * 100
        ax.hist(project_data, bins=20, alpha=0.6, label=project, color=colors[project], edgecolor='black')
    ax.set_xlabel('Acceptance Rate (%)', fontsize=12)
    ax.set_ylabel('Number of Developers', fontsize=12)
    ax.set_title('Distribution of Developer Acceptance Rates', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 1-2: バイオリンプロット
    ax = axes[0, 1]
    project_order = summary.sort_values('avg_acceptance_rate')['project'].tolist()
    violin_data = []
    labels = []
    for project in project_order:
        project_data = all_data[all_data['project'] == project]['acceptance_rate'] * 100
        violin_data.append(project_data)
        labels.append(project)

    parts = ax.violinplot(violin_data, vert=False, showmeans=True, showmedians=True)
    ax.set_yticks(range(1, len(labels) + 1))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Acceptance Rate (%)', fontsize=12)
    ax.set_title('Distribution by Project (Violin Plot)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # 1-3: 承諾率の分布（カテゴリ別）
    ax = axes[1, 0]
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels_bins = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']

    distribution_data = []
    for project in ['Qt', 'Android', 'Chromium', 'OpenStack']:
        project_data = all_data[all_data['project'] == project].copy()
        project_data['acceptance_rate_bin'] = pd.cut(
            project_data['acceptance_rate'],
            bins=bins,
            labels=labels_bins,
            include_lowest=True
        )
        dist = project_data['acceptance_rate_bin'].value_counts(normalize=True).sort_index() * 100
        distribution_data.append(dist)

    distribution_df = pd.DataFrame(distribution_data, index=['Qt', 'Android', 'Chromium', 'OpenStack'])
    distribution_df = distribution_df.reindex(['Qt', 'Android', 'OpenStack', 'Chromium'])

    x = np.arange(len(labels_bins))
    width = 0.2
    for i, project in enumerate(['Qt', 'Android', 'OpenStack', 'Chromium']):
        values = [distribution_df.loc[project, label] if label in distribution_df.columns else 0
                  for label in labels_bins]
        ax.bar(x + i * width, values, width, label=project, color=colors[project], alpha=0.8)

    ax.set_xlabel('Acceptance Rate Range', fontsize=12)
    ax.set_ylabel('Percentage of Developers (%)', fontsize=12)
    ax.set_title('Developer Distribution by Acceptance Rate Range', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(labels_bins)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 1-4: 平均承諾率の比較
    ax = axes[1, 1]
    summary_sorted = summary.sort_values('avg_acceptance_rate', ascending=True)
    bars = ax.barh(summary_sorted['project'], summary_sorted['avg_acceptance_rate'] * 100,
                   color=[colors[p] for p in summary_sorted['project']])
    ax.set_xlabel('Average Acceptance Rate (%)', fontsize=12)
    ax.set_title('Average Developer Acceptance Rate by Project', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    for i, (v, p) in enumerate(zip(summary_sorted['avg_acceptance_rate'] * 100, summary_sorted['project'])):
        ax.text(v + 1, i, f'{v:.2f}%', va='center', fontsize=10)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'acceptance_rate_distribution.png', dpi=300, bbox_inches='tight')
    print(f"保存: {output_dir / 'acceptance_rate_distribution.png'}")
    plt.close()

    # 図2: 承諾率 vs リクエスト数の関係
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for idx, project in enumerate(['Qt', 'Android', 'Chromium', 'OpenStack']):
        ax = axes[idx // 2, idx % 2]
        project_data = all_data[all_data['project'] == project]

        # リクエスト数でフィルタリング（最低2件以上）
        filtered_data = project_data[project_data['eval_request_count'] >= 2]

        scatter = ax.scatter(filtered_data['eval_request_count'],
                           filtered_data['acceptance_rate'] * 100,
                           s=50, alpha=0.6, color=colors[project], edgecolors='black', linewidth=0.5)

        ax.set_xlabel('Number of Review Requests', fontsize=11)
        ax.set_ylabel('Acceptance Rate (%)', fontsize=11)
        ax.set_title(f'{project} - Acceptance Rate vs Request Count', fontsize=12, fontweight='bold')
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)

        # 統計情報を追加
        correlation = filtered_data['eval_request_count'].corr(filtered_data['acceptance_rate'])
        ax.text(0.95, 0.05, f'Correlation: {correlation:.3f}',
               transform=ax.transAxes, ha='right', va='bottom',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'acceptance_vs_request_count.png', dpi=300, bbox_inches='tight')
    print(f"保存: {output_dir / 'acceptance_vs_request_count.png'}")
    plt.close()

    # 図3: 各プロジェクトのTop/Bottom開発者の比較
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for idx, project in enumerate(['Qt', 'Android', 'Chromium', 'OpenStack']):
        ax = axes[idx // 2, idx % 2]
        project_data = all_data[all_data['project'] == project].copy()

        # 最低5件以上のリクエストがある開発者に絞る
        filtered = project_data[project_data['eval_request_count'] >= 5].sort_values('acceptance_rate')

        if len(filtered) > 20:
            top_10 = filtered.tail(10)
            bottom_10 = filtered.head(10)
            combined = pd.concat([bottom_10, top_10])
        else:
            combined = filtered

        y_pos = np.arange(len(combined))
        bars = ax.barh(y_pos, combined['acceptance_rate'] * 100,
                      color=[colors[project] if rate >= 0.5 else '#95a5a6'
                            for rate in combined['acceptance_rate']])

        ax.set_yticks(y_pos)
        ax.set_yticklabels([email[:25] + '...' if len(email) > 25 else email
                           for email in combined['reviewer_email']], fontsize=8)
        ax.set_xlabel('Acceptance Rate (%)', fontsize=11)
        ax.set_title(f'{project} - Top/Bottom Developers (≥5 requests)', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'top_bottom_developers.png', dpi=300, bbox_inches='tight')
    print(f"保存: {output_dir / 'top_bottom_developers.png'}")
    plt.close()

    # 図4: 時系列パターンの分析（評価期間ごと）
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    period_order = {'0-3m': 0, '3-6m': 1, '6-9m': 2, '9-12m': 3}
    all_data['eval_period_num'] = all_data['eval_period'].map(period_order)

    for idx, project in enumerate(['Qt', 'Android', 'Chromium', 'OpenStack']):
        ax = axes[idx // 2, idx % 2]
        project_data = all_data[all_data['project'] == project]

        period_stats = project_data.groupby('eval_period_num').agg({
            'acceptance_rate': ['mean', 'std', 'count']
        }).reset_index()

        mean_vals = period_stats[('acceptance_rate', 'mean')] * 100
        std_vals = period_stats[('acceptance_rate', 'std')] * 100
        x_vals = period_stats['eval_period_num']

        ax.plot(x_vals, mean_vals, marker='o', color=colors[project],
               linewidth=2, markersize=8, label='Mean')
        ax.fill_between(x_vals, mean_vals - std_vals, mean_vals + std_vals,
                        alpha=0.3, color=colors[project])

        ax.set_xlabel('Evaluation Period', fontsize=11)
        ax.set_ylabel('Acceptance Rate (%)', fontsize=11)
        ax.set_title(f'{project} - Acceptance Rate Over Time', fontsize=12, fontweight='bold')
        ax.set_xticks(list(period_order.values()))
        ax.set_xticklabels(list(period_order.keys()))
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_dir / 'acceptance_temporal_trends.png', dpi=300, bbox_inches='tight')
    print(f"保存: {output_dir / 'acceptance_temporal_trends.png'}")
    plt.close()

    print(f"\n全ての可視化ファイルを保存しました: {output_dir}")

if __name__ == "__main__":
    create_visualizations()

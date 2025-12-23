"""
プロジェクト間のIRL統計を視覚化するスクリプト
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_visualizations():
    stats_dir = Path("/Users/kazuki-h/research/multiproject_research/results/irl_project_stats")
    output_dir = stats_dir / "visualizations"
    output_dir.mkdir(exist_ok=True)

    # サマリーを読み込み
    summary = pd.read_csv(stats_dir / "project_summary.csv")
    summary = summary.sort_values('developer_continuation_rate', ascending=True)

    # 色設定
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    project_colors = dict(zip(summary['project'], colors))

    # 図1: 開発者継続率の比較
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1-1: 開発者継続率
    ax = axes[0, 0]
    bars = ax.barh(summary['project'], summary['developer_continuation_rate'] * 100,
                   color=[project_colors[p] for p in summary['project']])
    ax.set_xlabel('Developer Continuation Rate (%)', fontsize=12)
    ax.set_title('1. Developer Continuation Rate by Project', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    for i, (v, p) in enumerate(zip(summary['developer_continuation_rate'] * 100, summary['project'])):
        ax.text(v + 1, i, f'{v:.2f}%', va='center', fontsize=10)
    ax.grid(axis='x', alpha=0.3)

    # 1-2: レビュー承諾率（評価期間の全体）
    ax = axes[0, 1]
    bars = ax.barh(summary['project'], summary['overall_eval_acceptance_rate'] * 100,
                   color=[project_colors[p] for p in summary['project']])
    ax.set_xlabel('Review Acceptance Rate (%)', fontsize=12)
    ax.set_title('2. Review Acceptance Rate (Eval Period)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    for i, (v, p) in enumerate(zip(summary['overall_eval_acceptance_rate'] * 100, summary['project'])):
        ax.text(v + 1, i, f'{v:.2f}%', va='center', fontsize=10)
    ax.grid(axis='x', alpha=0.3)

    # 1-3: 総開発者数とドロップアウト数
    ax = axes[1, 0]
    x = range(len(summary))
    width = 0.35
    ax.bar(x, summary['continuing_developers'], width, label='Continuing',
           color=[project_colors[p] for p in summary['project']], alpha=0.8)
    ax.bar(x, summary['dropout_developers'], width, bottom=summary['continuing_developers'],
           label='Dropout', color='gray', alpha=0.6)
    ax.set_ylabel('Number of Developers', fontsize=12)
    ax.set_title('3. Developer Counts (Continuing vs Dropout)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(summary['project'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 1-4: 開発者あたりレビュー数
    ax = axes[1, 1]
    avg_reviews = summary['total_eval_requests'] / summary['total_developers']
    summary_sorted = summary.copy()
    summary_sorted['avg_reviews'] = avg_reviews
    summary_sorted = summary_sorted.sort_values('avg_reviews', ascending=True)
    bars = ax.barh(summary_sorted['project'], summary_sorted['avg_reviews'],
                   color=[project_colors[p] for p in summary_sorted['project']])
    ax.set_xlabel('Reviews per Developer', fontsize=12)
    ax.set_title('4. Average Reviews per Developer', fontsize=14, fontweight='bold')
    for i, (v, p) in enumerate(zip(summary_sorted['avg_reviews'], summary_sorted['project'])):
        ax.text(v + 0.5, i, f'{v:.1f}', va='center', fontsize=10)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'project_comparison_overview.png', dpi=300, bbox_inches='tight')
    print(f"保存: {output_dir / 'project_comparison_overview.png'}")
    plt.close()

    # 図2: 承諾率の比較（過去 vs 評価期間）
    fig, ax = plt.subplots(figsize=(12, 8))
    x = range(len(summary))
    width = 0.35
    ax.bar([i - width/2 for i in x], summary['avg_history_acceptance_rate'] * 100, width,
           label='History Period', color='#95a5a6', alpha=0.8)
    ax.bar([i + width/2 for i in x], summary['avg_eval_acceptance_rate'] * 100, width,
           label='Eval Period', color='#3498db', alpha=0.8)
    ax.set_ylabel('Acceptance Rate (%)', fontsize=12)
    ax.set_title('Review Acceptance Rate: History vs Eval Period', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(summary['project'])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_dir / 'acceptance_rate_comparison.png', dpi=300, bbox_inches='tight')
    print(f"保存: {output_dir / 'acceptance_rate_comparison.png'}")
    plt.close()

    # 図3: 散布図 - 継続率 vs 承諾率
    fig, ax = plt.subplots(figsize=(10, 8))
    for project in summary['project']:
        data = summary[summary['project'] == project]
        ax.scatter(data['overall_eval_acceptance_rate'] * 100,
                  data['developer_continuation_rate'] * 100,
                  s=data['total_developers'] / 5,
                  color=project_colors[project],
                  label=project,
                  alpha=0.7,
                  edgecolors='black',
                  linewidth=1.5)

    ax.set_xlabel('Review Acceptance Rate (%)', fontsize=12)
    ax.set_ylabel('Developer Continuation Rate (%)', fontsize=12)
    ax.set_title('Continuation Rate vs Acceptance Rate\n(Bubble size = # of developers)',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(50, 80)
    ax.set_ylim(65, 95)

    plt.tight_layout()
    plt.savefig(output_dir / 'continuation_vs_acceptance.png', dpi=300, bbox_inches='tight')
    print(f"保存: {output_dir / 'continuation_vs_acceptance.png'}")
    plt.close()

    # 図4: 時系列での変化（各プロジェクトの全データを使用）
    all_data = pd.read_csv(stats_dir / "all_projects_stats.csv")

    # 訓練期間を数値に変換
    period_order = {'0-3m': 0, '3-6m': 1, '6-9m': 2, '9-12m': 3}
    all_data['train_period_num'] = all_data['train_period'].map(period_order)
    all_data['eval_period_num'] = all_data['eval_period'].map(period_order)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 4-1: 継続率の時系列変化（訓練期間ごと）
    ax = axes[0, 0]
    for project in summary['project']:
        project_data = all_data[all_data['project'] == project].groupby('train_period_num').agg({
            'developer_continuation_rate': 'mean'
        }).reset_index()
        ax.plot(project_data['train_period_num'], project_data['developer_continuation_rate'] * 100,
               marker='o', label=project, color=project_colors[project], linewidth=2, markersize=8)
    ax.set_xlabel('Training Period', fontsize=12)
    ax.set_ylabel('Developer Continuation Rate (%)', fontsize=12)
    ax.set_title('Continuation Rate by Training Period', fontsize=14, fontweight='bold')
    ax.set_xticks(list(period_order.values()))
    ax.set_xticklabels(list(period_order.keys()))
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4-2: 承諾率の時系列変化（訓練期間ごと）
    ax = axes[0, 1]
    for project in summary['project']:
        project_data = all_data[all_data['project'] == project].groupby('train_period_num').agg({
            'overall_eval_acceptance_rate': 'mean'
        }).reset_index()
        ax.plot(project_data['train_period_num'], project_data['overall_eval_acceptance_rate'] * 100,
               marker='s', label=project, color=project_colors[project], linewidth=2, markersize=8)
    ax.set_xlabel('Training Period', fontsize=12)
    ax.set_ylabel('Review Acceptance Rate (%)', fontsize=12)
    ax.set_title('Acceptance Rate by Training Period', fontsize=14, fontweight='bold')
    ax.set_xticks(list(period_order.values()))
    ax.set_xticklabels(list(period_order.keys()))
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4-3: 継続率の時系列変化（評価期間ごと）
    ax = axes[1, 0]
    for project in summary['project']:
        project_data = all_data[all_data['project'] == project].groupby('eval_period_num').agg({
            'developer_continuation_rate': 'mean'
        }).reset_index()
        ax.plot(project_data['eval_period_num'], project_data['developer_continuation_rate'] * 100,
               marker='o', label=project, color=project_colors[project], linewidth=2, markersize=8)
    ax.set_xlabel('Evaluation Period', fontsize=12)
    ax.set_ylabel('Developer Continuation Rate (%)', fontsize=12)
    ax.set_title('Continuation Rate by Evaluation Period', fontsize=14, fontweight='bold')
    ax.set_xticks(list(period_order.values()))
    ax.set_xticklabels(list(period_order.keys()))
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4-4: 承諾率の時系列変化（評価期間ごと）
    ax = axes[1, 1]
    for project in summary['project']:
        project_data = all_data[all_data['project'] == project].groupby('eval_period_num').agg({
            'overall_eval_acceptance_rate': 'mean'
        }).reset_index()
        ax.plot(project_data['eval_period_num'], project_data['overall_eval_acceptance_rate'] * 100,
               marker='s', label=project, color=project_colors[project], linewidth=2, markersize=8)
    ax.set_xlabel('Evaluation Period', fontsize=12)
    ax.set_ylabel('Review Acceptance Rate (%)', fontsize=12)
    ax.set_title('Acceptance Rate by Evaluation Period', fontsize=14, fontweight='bold')
    ax.set_xticks(list(period_order.values()))
    ax.set_xticklabels(list(period_order.keys()))
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'temporal_trends.png', dpi=300, bbox_inches='tight')
    print(f"保存: {output_dir / 'temporal_trends.png'}")
    plt.close()

    print(f"\n全ての可視化ファイルを保存しました: {output_dir}")

if __name__ == "__main__":
    create_visualizations()

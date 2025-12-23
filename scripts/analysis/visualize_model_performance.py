"""
IRLモデルの性能を視覚化するスクリプト
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
    data_dir = Path("/Users/kazuki-h/research/multiproject_research/results/irl_model_performance")
    output_dir = data_dir / "visualizations"
    output_dir.mkdir(exist_ok=True)

    # データを読み込み
    all_data = pd.read_csv(data_dir / "all_projects_model_performance.csv")
    summary = pd.read_csv(data_dir / "project_performance_summary.csv")

    # 色設定
    colors = {'Qt': '#FF6B6B', 'Android': '#4ECDC4', 'Chromium': '#FFA07A', 'OpenStack': '#45B7D1'}

    # 図1: プロジェクト別性能比較
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1-1: AUC-ROC
    ax = axes[0, 0]
    summary_sorted = summary.sort_values('auc_roc', ascending=True)
    bars = ax.barh(summary_sorted['project'], summary_sorted['auc_roc'],
                   color=[colors[p] for p in summary_sorted['project']])
    ax.set_xlabel('AUC-ROC', fontsize=12)
    ax.set_title('Average AUC-ROC by Project', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    for i, (v, p) in enumerate(zip(summary_sorted['auc_roc'], summary_sorted['project'])):
        ax.text(v + 0.02, i, f'{v:.4f}', va='center', fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    ax.legend()

    # 1-2: AUC-PR
    ax = axes[0, 1]
    summary_sorted = summary.sort_values('auc_pr', ascending=True)
    bars = ax.barh(summary_sorted['project'], summary_sorted['auc_pr'],
                   color=[colors[p] for p in summary_sorted['project']])
    ax.set_xlabel('AUC-PR', fontsize=12)
    ax.set_title('Average AUC-PR by Project', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    for i, (v, p) in enumerate(zip(summary_sorted['auc_pr'], summary_sorted['project'])):
        ax.text(v + 0.02, i, f'{v:.4f}', va='center', fontsize=10)
    ax.grid(axis='x', alpha=0.3)

    # 1-3: F1 Score
    ax = axes[1, 0]
    summary_sorted = summary.sort_values('f1_score', ascending=True)
    bars = ax.barh(summary_sorted['project'], summary_sorted['f1_score'],
                   color=[colors[p] for p in summary_sorted['project']])
    ax.set_xlabel('F1 Score', fontsize=12)
    ax.set_title('Average F1 Score by Project', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    for i, (v, p) in enumerate(zip(summary_sorted['f1_score'], summary_sorted['project'])):
        ax.text(v + 0.02, i, f'{v:.4f}', va='center', fontsize=10)
    ax.grid(axis='x', alpha=0.3)

    # 1-4: Precision vs Recall
    ax = axes[1, 1]
    x = np.arange(len(summary))
    width = 0.35
    summary_plot = summary.set_index('project').loc[['Qt', 'Android', 'OpenStack', 'Chromium']]
    ax.bar([i - width/2 for i in x], summary_plot['precision'], width,
           label='Precision', color='#3498db', alpha=0.8)
    ax.bar([i + width/2 for i in x], summary_plot['recall'], width,
           label='Recall', color='#e74c3c', alpha=0.8)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Precision vs Recall by Project', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(summary_plot.index)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(output_dir / 'model_performance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"保存: {output_dir / 'model_performance_comparison.png'}")
    plt.close()

    # 図2: 全メトリクスの比較（レーダーチャート風）
    fig, ax = plt.subplots(figsize=(12, 10))

    metrics = ['auc_roc', 'auc_pr', 'f1_score', 'precision', 'recall']
    x = np.arange(len(metrics))
    width = 0.2

    for i, project in enumerate(['Qt', 'Android', 'OpenStack', 'Chromium']):
        project_data = summary[summary['project'] == project]
        values = [project_data[metric].values[0] for metric in metrics]
        ax.bar(x + i * width, values, width, label=project, color=colors[project], alpha=0.8)

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('All Metrics Comparison by Project', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(['AUC-ROC', 'AUC-PR', 'F1', 'Precision', 'Recall'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(output_dir / 'all_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"保存: {output_dir / 'all_metrics_comparison.png'}")
    plt.close()

    # 図3: 時系列パターン（評価期間ごと）
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    period_order = {'0-3m': 0, '3-6m': 1, '6-9m': 2, '9-12m': 3}
    all_data['eval_period_num'] = all_data['eval_period'].map(period_order)

    metrics_plot = [
        ('auc_roc', 'AUC-ROC', axes[0, 0]),
        ('auc_pr', 'AUC-PR', axes[0, 1]),
        ('f1_score', 'F1 Score', axes[1, 0]),
        ('precision', 'Precision', axes[1, 1])
    ]

    for metric, title, ax in metrics_plot:
        for project in ['Qt', 'Android', 'Chromium', 'OpenStack']:
            project_data = all_data[all_data['project'] == project].groupby('eval_period_num')[metric].mean()
            ax.plot(project_data.index, project_data.values, marker='o',
                   label=project, color=colors[project], linewidth=2, markersize=8)

        ax.set_xlabel('Evaluation Period', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f'{title} by Evaluation Period', fontsize=12, fontweight='bold')
        ax.set_xticks(list(period_order.values()))
        ax.set_xticklabels(list(period_order.keys()))
        ax.legend()
        ax.grid(True, alpha=0.3)
        if metric == 'auc_roc':
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_temporal_trends.png', dpi=300, bbox_inches='tight')
    print(f"保存: {output_dir / 'performance_temporal_trends.png'}")
    plt.close()

    # 図4: 分布の比較（ボックスプロット）
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    metrics_box = [
        ('auc_roc', 'AUC-ROC Distribution', axes[0, 0]),
        ('auc_pr', 'AUC-PR Distribution', axes[0, 1]),
        ('f1_score', 'F1 Score Distribution', axes[1, 0]),
        ('precision', 'Precision Distribution', axes[1, 1])
    ]

    for metric, title, ax in metrics_box:
        data_to_plot = []
        labels = []
        for project in ['Qt', 'Android', 'OpenStack', 'Chromium']:
            project_data = all_data[all_data['project'] == project][metric].dropna()
            data_to_plot.append(project_data)
            labels.append(project)

        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch, project in zip(bp['boxes'], labels):
            patch.set_facecolor(colors[project])
            patch.set_alpha(0.7)

        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        if metric == 'auc_roc':
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_distribution.png', dpi=300, bbox_inches='tight')
    print(f"保存: {output_dir / 'performance_distribution.png'}")
    plt.close()

    # 図5: 散布図 - AUC-ROC vs F1 Score
    fig, ax = plt.subplots(figsize=(10, 8))

    for project in ['Qt', 'Android', 'Chromium', 'OpenStack']:
        project_data = all_data[all_data['project'] == project]
        ax.scatter(project_data['auc_roc'], project_data['f1_score'],
                  s=100, alpha=0.6, color=colors[project], label=project,
                  edgecolors='black', linewidth=1)

    ax.set_xlabel('AUC-ROC', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('AUC-ROC vs F1 Score', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'auc_vs_f1.png', dpi=300, bbox_inches='tight')
    print(f"保存: {output_dir / 'auc_vs_f1.png'}")
    plt.close()

    print(f"\n全ての可視化ファイルを保存しました: {output_dir}")

if __name__ == "__main__":
    create_visualizations()

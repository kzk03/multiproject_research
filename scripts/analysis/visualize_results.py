import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

# スタイル設定
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_metrics(base_path: str, train_period: str, eval_period: str) -> Dict:
    """メトリクスファイルを読み込む"""
    metrics_path = Path(base_path) / f"train_{train_period}" / f"eval_{eval_period}" / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None

def collect_all_results(results_dir: Path) -> Dict:
    """全ての結果を収集"""
    experiments = {
        'Single Project (Nova)': results_dir / "review_continuation_cross_eval_nova",
        'Multi-Project (No OS)': results_dir / "cross_temporal_openstack_multiproject_2020_2024",
        'Multi-Project (2x OS)': results_dir / "cross_temporal_openstack_multiproject_os2x",
        'Multi-Project (3x OS)': results_dir / "cross_temporal_openstack_multiproject_os3x",
    }

    train_periods = ['0-3m', '3-6m', '6-9m', '9-12m']
    eval_periods = ['0-3m', '3-6m', '6-9m', '9-12m']

    all_results = {}

    for exp_name, path in experiments.items():
        if not path.exists():
            continue

        all_results[exp_name] = {}
        for train in train_periods:
            for eval_p in eval_periods:
                metrics = load_metrics(str(path), train, eval_p)
                if metrics:
                    key = f"{train}→{eval_p}"
                    all_results[exp_name][key] = metrics

    return all_results

def plot_performance_comparison(all_results: Dict, output_dir: Path):
    """性能比較グラフ"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Performance Comparison Across Different Settings', fontsize=16, fontweight='bold')

    metrics_to_plot = ['f1_score', 'precision', 'recall', 'auc_roc']
    metric_names = ['F1 Score', 'Precision', 'Recall', 'AUC-ROC']

    for idx, (metric, metric_name) in enumerate(zip(metrics_to_plot, metric_names)):
        ax = axes[idx // 2, idx % 2]

        # 同一期間と未来期間のデータを準備
        same_period_data = []
        future_period_data = []
        labels = []

        for exp_name in all_results.keys():
            labels.append(exp_name.replace('Multi-Project', 'MP').replace('Single Project', 'SP'))

            # 同一期間
            same_values = []
            future_values = []

            for key, metrics in all_results[exp_name].items():
                train, eval_p = key.split('→')
                if metric in metrics:
                    if train == eval_p:
                        same_values.append(metrics[metric])
                    elif int(eval_p.split('-')[0]) > int(train.split('-')[0]):
                        future_values.append(metrics[metric])

            same_period_data.append(np.mean(same_values) if same_values else 0)
            future_period_data.append(np.mean(future_values) if future_values else 0)

        x = np.arange(len(labels))
        width = 0.35

        bars1 = ax.bar(x - width/2, same_period_data, width, label='Same Period', alpha=0.8)
        bars2 = ax.bar(x + width/2, future_period_data, width, label='Future Period', alpha=0.8)

        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f'{metric_name} Comparison', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 値をバーの上に表示
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"保存しました: {output_dir / 'performance_comparison.png'}")

def plot_temporal_heatmap(all_results: Dict, output_dir: Path):
    """時系列ヒートマップ"""
    train_periods = ['0-3m', '3-6m', '6-9m', '9-12m']
    eval_periods = ['0-3m', '3-6m', '6-9m', '9-12m']

    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    fig.suptitle('F1 Score Heatmaps: Training Period × Evaluation Period', fontsize=16, fontweight='bold')

    for idx, exp_name in enumerate(all_results.keys()):
        ax = axes[idx // 2, idx % 2]

        # F1スコア行列を作成
        f1_matrix = np.zeros((len(train_periods), len(eval_periods)))

        for i, train in enumerate(train_periods):
            for j, eval_p in enumerate(eval_periods):
                key = f"{train}→{eval_p}"
                if key in all_results[exp_name] and 'f1_score' in all_results[exp_name][key]:
                    f1_matrix[i, j] = all_results[exp_name][key]['f1_score']
                else:
                    f1_matrix[i, j] = np.nan

        # ヒートマップ描画
        im = ax.imshow(f1_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

        # ラベル設定
        ax.set_xticks(np.arange(len(eval_periods)))
        ax.set_yticks(np.arange(len(train_periods)))
        ax.set_xticklabels(eval_periods)
        ax.set_yticklabels(train_periods)

        ax.set_xlabel('Evaluation Period', fontsize=11)
        ax.set_ylabel('Training Period', fontsize=11)
        ax.set_title(exp_name, fontsize=12, fontweight='bold')

        # 値をセルに表示
        for i in range(len(train_periods)):
            for j in range(len(eval_periods)):
                if not np.isnan(f1_matrix[i, j]):
                    text = ax.text(j, i, f'{f1_matrix[i, j]:.3f}',
                                 ha="center", va="center", color="black", fontsize=10)

        # カラーバー
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(output_dir / 'temporal_heatmaps.png', dpi=300, bbox_inches='tight')
    print(f"保存しました: {output_dir / 'temporal_heatmaps.png'}")

def plot_precision_recall_curve(all_results: Dict, output_dir: Path):
    """Precision-Recallバランスの比較"""
    fig, ax = plt.subplots(figsize=(12, 8))

    markers = ['o', 's', '^', 'D']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, (exp_name, color, marker) in enumerate(zip(all_results.keys(), colors, markers)):
        precisions = []
        recalls = []

        for key, metrics in all_results[exp_name].items():
            if 'precision' in metrics and 'recall' in metrics:
                precisions.append(metrics['precision'])
                recalls.append(metrics['recall'])

        if precisions and recalls:
            # 平均をプロット
            mean_prec = np.mean(precisions)
            mean_rec = np.mean(recalls)
            std_prec = np.std(precisions)
            std_rec = np.std(recalls)

            ax.scatter(mean_rec, mean_prec, s=200, alpha=0.7,
                      color=color, marker=marker,
                      label=exp_name, edgecolors='black', linewidth=2)

            # エラーバー
            ax.errorbar(mean_rec, mean_prec,
                       xerr=std_rec, yerr=std_prec,
                       fmt='none', color=color, alpha=0.5, capsize=5)

            # 個別のポイント
            ax.scatter(recalls, precisions, s=30, alpha=0.3, color=color)

    ax.set_xlabel('Recall', fontsize=14, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=14, fontweight='bold')
    ax.set_title('Precision-Recall Trade-off', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_dir / 'precision_recall_tradeoff.png', dpi=300, bbox_inches='tight')
    print(f"保存しました: {output_dir / 'precision_recall_tradeoff.png'}")

def plot_oversampling_effect(all_results: Dict, output_dir: Path):
    """オーバーサンプリング効果の可視化"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 複数プロジェクトの結果のみを抽出
    multiproject_results = {k: v for k, v in all_results.items() if 'Multi-Project' in k}

    settings = ['No OS', '2x OS', '3x OS']
    colors_map = {
        'Multi-Project (No OS)': 'No OS',
        'Multi-Project (2x OS)': '2x OS',
        'Multi-Project (3x OS)': '3x OS'
    }

    # 同一期間での性能
    ax1 = axes[0]
    metrics = ['F1 Score', 'Precision', 'Recall', 'AUC-ROC']
    metric_keys = ['f1_score', 'precision', 'recall', 'auc_roc']

    x = np.arange(len(settings))
    width = 0.2

    for idx, (metric_name, metric_key) in enumerate(zip(metrics, metric_keys)):
        values = []
        for setting in settings:
            exp_name = f'Multi-Project ({setting})'
            if exp_name in multiproject_results:
                same_period_values = []
                for key, m in multiproject_results[exp_name].items():
                    train, eval_p = key.split('→')
                    if train == eval_p and metric_key in m:
                        same_period_values.append(m[metric_key])
                values.append(np.mean(same_period_values) if same_period_values else 0)
            else:
                values.append(0)

        ax1.bar(x + idx * width, values, width, label=metric_name, alpha=0.8)

    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Same Period Performance vs Oversampling', fontsize=13, fontweight='bold')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(settings)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 未来期間での性能
    ax2 = axes[1]

    for idx, (metric_name, metric_key) in enumerate(zip(metrics, metric_keys)):
        values = []
        for setting in settings:
            exp_name = f'Multi-Project ({setting})'
            if exp_name in multiproject_results:
                future_period_values = []
                for key, m in multiproject_results[exp_name].items():
                    train, eval_p = key.split('→')
                    if int(eval_p.split('-')[0]) > int(train.split('-')[0]) and metric_key in m:
                        future_period_values.append(m[metric_key])
                values.append(np.mean(future_period_values) if future_period_values else 0)
            else:
                values.append(0)

        ax2.bar(x + idx * width, values, width, label=metric_name, alpha=0.8)

    ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax2.set_title('Future Period Performance vs Oversampling', fontsize=13, fontweight='bold')
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(settings)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'oversampling_effect.png', dpi=300, bbox_inches='tight')
    print(f"保存しました: {output_dir / 'oversampling_effect.png'}")

def plot_temporal_stability(all_results: Dict, output_dir: Path):
    """時系列での性能安定性"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Temporal Performance Stability', fontsize=16, fontweight='bold')

    train_periods = ['0-3m', '3-6m', '6-9m', '9-12m']

    for idx, exp_name in enumerate(all_results.keys()):
        ax = axes[idx // 2, idx % 2]

        for train in train_periods:
            f1_scores = []
            eval_labels = []

            for eval_p in ['0-3m', '3-6m', '6-9m', '9-12m']:
                key = f"{train}→{eval_p}"
                if key in all_results[exp_name] and 'f1_score' in all_results[exp_name][key]:
                    f1_scores.append(all_results[exp_name][key]['f1_score'])
                    eval_labels.append(eval_p)

            if f1_scores:
                ax.plot(range(len(f1_scores)), f1_scores, marker='o',
                       label=f'Train: {train}', linewidth=2, markersize=8)

        ax.set_xlabel('Evaluation Period', fontsize=11)
        ax.set_ylabel('F1 Score', fontsize=11)
        ax.set_title(exp_name, fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(['0-3m', '3-6m', '6-9m', '9-12m'])))
        ax.set_xticklabels(['0-3m', '3-6m', '6-9m', '9-12m'])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'temporal_stability.png', dpi=300, bbox_inches='tight')
    print(f"保存しました: {output_dir / 'temporal_stability.png'}")

def main():
    results_dir = Path("/Users/kazuki-h/research/multiproject_research/results")
    output_dir = Path("/Users/kazuki-h/research/multiproject_research/docs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("結果を収集中...")
    all_results = collect_all_results(results_dir)

    print("\n可視化を生成中...")

    print("\n1. 性能比較グラフ...")
    plot_performance_comparison(all_results, output_dir)

    print("\n2. 時系列ヒートマップ...")
    plot_temporal_heatmap(all_results, output_dir)

    print("\n3. Precision-Recall曲線...")
    plot_precision_recall_curve(all_results, output_dir)

    print("\n4. オーバーサンプリング効果...")
    plot_oversampling_effect(all_results, output_dir)

    print("\n5. 時系列安定性...")
    plot_temporal_stability(all_results, output_dir)

    print(f"\n全ての可視化が完了しました: {output_dir}")

if __name__ == "__main__":
    main()

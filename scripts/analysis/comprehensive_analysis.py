"""
包括的な分析スクリプト
- 特徴量の違いを考慮した分析
- 詳細な可視化
- 統計的検定
"""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from scipy import stats
from scipy.stats import mannwhitneyu, wilcoxon

# スタイル設定
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100

def load_all_results(results_dir: Path) -> Dict:
    """全ての結果を読み込む"""
    experiments = {
        'Single Project (Nova)': {
            'path': results_dir / "review_continuation_cross_eval_nova",
            'state_dim': 10,  # 単一プロジェクト: 10次元
            'action_dim': 4,  # 単一プロジェクト: 4次元
            'features': 'Basic features only'
        },
        'Multi-Project (No OS)': {
            'path': results_dir / "cross_temporal_openstack_multiproject_2020_2024",
            'state_dim': 14,  # 複数プロジェクト: 14次元
            'action_dim': 5,  # 複数プロジェクト: 5次元
            'features': 'Basic + Multi-project features'
        },
        'Multi-Project (2x OS)': {
            'path': results_dir / "cross_temporal_openstack_multiproject_os2x",
            'state_dim': 14,
            'action_dim': 5,
            'features': 'Basic + Multi-project features'
        },
        'Multi-Project (3x OS)': {
            'path': results_dir / "cross_temporal_openstack_multiproject_os3x",
            'state_dim': 14,
            'action_dim': 5,
            'features': 'Basic + Multi-project features'
        },
    }

    train_periods = ['0-3m', '3-6m', '6-9m', '9-12m']
    eval_periods = ['0-3m', '3-6m', '6-9m', '9-12m']

    results = {}

    for exp_name, exp_info in experiments.items():
        path = exp_info['path']
        if not path.exists():
            continue

        exp_results = {
            'metrics': [],
            'predictions': [],
            'state_dim': exp_info['state_dim'],
            'action_dim': exp_info['action_dim'],
            'features': exp_info['features']
        }

        for train in train_periods:
            for eval_p in eval_periods:
                # メトリクス
                metrics_file = path / f"train_{train}" / f"eval_{eval_p}" / "metrics.json"
                if metrics_file.exists():
                    with open(metrics_file) as f:
                        metrics = json.load(f)
                    metrics['train_period'] = train
                    metrics['eval_period'] = eval_p
                    exp_results['metrics'].append(metrics)

                # 予測
                pred_file = path / f"train_{train}" / f"eval_{eval_p}" / "predictions.csv"
                if pred_file.exists():
                    df = pd.read_csv(pred_file)
                    df['train_period'] = train
                    df['eval_period'] = eval_p
                    exp_results['predictions'].append(df)

        if exp_results['predictions']:
            exp_results['predictions_df'] = pd.concat(exp_results['predictions'], ignore_index=True)

        results[exp_name] = exp_results

    return results

def plot_feature_comparison(results: Dict, output_dir: Path):
    """特徴量次元の比較"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Feature Dimensions Comparison', fontsize=16, fontweight='bold')

    exp_names = list(results.keys())
    state_dims = [results[name]['state_dim'] for name in exp_names]
    action_dims = [results[name]['action_dim'] for name in exp_names]

    # 状態次元
    bars1 = ax1.barh(range(len(exp_names)), state_dims, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax1.set_yticks(range(len(exp_names)))
    ax1.set_yticklabels([name.replace('Multi-Project', 'MP') for name in exp_names])
    ax1.set_xlabel('State Dimension', fontweight='bold')
    ax1.set_title('State Features (Reviewer State)', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')

    # 値をバーに表示
    for i, (bar, dim) in enumerate(zip(bars1, state_dims)):
        feature_type = 'Basic (10)' if dim == 10 else 'Basic (10) +\nMulti-Project (4)'
        ax1.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                f'{dim} dims\n{feature_type}',
                va='center', fontsize=9)

    # 行動次元
    bars2 = ax2.barh(range(len(exp_names)), action_dims, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax2.set_yticks(range(len(exp_names)))
    ax2.set_yticklabels([name.replace('Multi-Project', 'MP') for name in exp_names])
    ax2.set_xlabel('Action Dimension', fontweight='bold')
    ax2.set_title('Action Features (Review Activity)', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')

    for i, (bar, dim) in enumerate(zip(bars2, action_dims)):
        feature_type = 'Basic (4)' if dim == 4 else 'Basic (4) +\nCross-Project (1)'
        ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f'{dim} dims\n{feature_type}',
                va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'feature_dimensions.png', dpi=300, bbox_inches='tight')
    print(f"保存: {output_dir / 'feature_dimensions.png'}")

def plot_performance_vs_features(results: Dict, output_dir: Path):
    """特徴量次元と性能の関係"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Performance vs Feature Dimensions', fontsize=16, fontweight='bold')

    metrics_to_plot = ['f1_score', 'precision', 'recall', 'auc_roc']
    metric_names = ['F1 Score', 'Precision', 'Recall', 'AUC-ROC']

    for idx, (metric, metric_name) in enumerate(zip(metrics_to_plot, metric_names)):
        ax = axes[idx // 2, idx % 2]

        # データ収集
        data = []
        for exp_name, exp_data in results.items():
            state_dim = exp_data['state_dim']
            total_dim = state_dim + exp_data['action_dim']

            for m in exp_data['metrics']:
                if metric in m:
                    data.append({
                        'experiment': exp_name.replace('Multi-Project', 'MP'),
                        'total_features': total_dim,
                        'state_dim': state_dim,
                        'value': m[metric]
                    })

        df = pd.DataFrame(data)

        # グループごとの平均をプロット
        for exp in df['experiment'].unique():
            subset = df[df['experiment'] == exp]
            total_features = subset['total_features'].iloc[0]
            mean_val = subset['value'].mean()
            std_val = subset['value'].std()

            # 色分け: 10次元 vs 14次元
            color = '#FF6B6B' if subset['state_dim'].iloc[0] == 10 else '#4ECDC4'
            marker = 'o' if subset['state_dim'].iloc[0] == 10 else 's'

            ax.errorbar(total_features, mean_val, yerr=std_val,
                       marker=marker, markersize=10, capsize=5,
                       label=exp, color=color, linewidth=2, alpha=0.7)

        ax.set_xlabel('Total Feature Dimensions (State + Action)', fontweight='bold')
        ax.set_ylabel(metric_name, fontweight='bold')
        ax.set_title(metric_name, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_vs_features.png', dpi=300, bbox_inches='tight')
    print(f"保存: {output_dir / 'performance_vs_features.png'}")

def statistical_comparison(results: Dict, output_dir: Path):
    """統計的比較分析"""
    print("\n" + "="*100)
    print("統計的比較分析")
    print("="*100)

    # Single Project vs Multi-Project (No OS) の比較
    single_metrics = results['Single Project (Nova)']['metrics']
    multi_metrics = results['Multi-Project (No OS)']['metrics']

    print("\n【Single Project vs Multi-Project (No OS)】")
    print("-"*100)

    for metric in ['f1_score', 'precision', 'recall', 'auc_roc']:
        single_vals = [m[metric] for m in single_metrics if metric in m]
        multi_vals = [m[metric] for m in multi_metrics if metric in m]

        # Mann-Whitney U検定（対応なし）
        stat, p_value = mannwhitneyu(single_vals, multi_vals, alternative='two-sided')

        print(f"\n{metric.upper()}:")
        print(f"  Single Project: {np.mean(single_vals):.4f} ± {np.std(single_vals):.4f}")
        print(f"  Multi-Project:  {np.mean(multi_vals):.4f} ± {np.std(multi_vals):.4f}")
        print(f"  Mann-Whitney U: statistic={stat:.2f}, p-value={p_value:.4f}")
        if p_value < 0.05:
            print(f"  → 有意差あり (p < 0.05)")
        else:
            print(f"  → 有意差なし (p >= 0.05)")

    # オーバーサンプリング間の比較
    print("\n\n【Multi-Project: Oversampling Comparison】")
    print("-"*100)

    os_experiments = ['Multi-Project (No OS)', 'Multi-Project (2x OS)', 'Multi-Project (3x OS)']

    for metric in ['f1_score', 'auc_roc']:
        print(f"\n{metric.upper()}:")

        for i, exp1 in enumerate(os_experiments):
            for exp2 in os_experiments[i+1:]:
                vals1 = [m[metric] for m in results[exp1]['metrics'] if metric in m]
                vals2 = [m[metric] for m in results[exp2]['metrics'] if metric in m]

                stat, p_value = mannwhitneyu(vals1, vals2, alternative='two-sided')

                print(f"  {exp1.split('(')[1][:-1]} vs {exp2.split('(')[1][:-1]}:")
                print(f"    Mean: {np.mean(vals1):.4f} vs {np.mean(vals2):.4f}")
                print(f"    p-value: {p_value:.4f}")

def plot_prediction_distribution(results: Dict, output_dir: Path):
    """予測確率の分布比較"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Prediction Probability Distribution by Experiment', fontsize=16, fontweight='bold')

    for idx, (exp_name, exp_data) in enumerate(results.items()):
        ax = axes[idx // 2, idx % 2]

        if 'predictions_df' not in exp_data:
            continue

        df = exp_data['predictions_df']

        # 真のラベル別に分布をプロット
        accepted = df[df['true_label'] == 1]['predicted_prob']
        rejected = df[df['true_label'] == 0]['predicted_prob']

        ax.hist(accepted, bins=30, alpha=0.6, label=f'Accepted (n={len(accepted)})',
               color='green', edgecolor='black')
        ax.hist(rejected, bins=30, alpha=0.6, label=f'Rejected (n={len(rejected)})',
               color='red', edgecolor='black')

        # 閾値を表示（最初のメトリクスから取得）
        if exp_data['metrics']:
            threshold = exp_data['metrics'][0].get('optimal_threshold', 0.5)
            ax.axvline(threshold, color='blue', linestyle='--', linewidth=2,
                      label=f'Threshold: {threshold:.3f}')

        ax.set_xlabel('Predicted Probability', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title(f"{exp_name.replace('Multi-Project', 'MP')}\n"
                    f"({exp_data['state_dim']+exp_data['action_dim']} total features)",
                    fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_distributions.png', dpi=300, bbox_inches='tight')
    print(f"保存: {output_dir / 'prediction_distributions.png'}")

def plot_confusion_matrices(results: Dict, output_dir: Path):
    """混同行列の可視化"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Confusion Matrices (Aggregated)', fontsize=16, fontweight='bold')

    for idx, (exp_name, exp_data) in enumerate(results.items()):
        ax = axes[idx // 2, idx % 2]

        if 'predictions_df' not in exp_data:
            continue

        df = exp_data['predictions_df']

        # 混同行列を計算
        y_true = df['true_label'].values
        y_pred = df['predicted_binary'].values

        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()

        cm = np.array([[tn, fp], [fn, tp]])

        # 正規化（行ごと）
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # ヒートマップ
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                   ax=ax, cbar_kws={'label': 'Proportion'},
                   xticklabels=['Predicted\nRejected', 'Predicted\nAccepted'],
                   yticklabels=['Actual\nRejected', 'Actual\nAccepted'])

        # 実数を追加表示
        for i in range(2):
            for j in range(2):
                ax.text(j+0.5, i+0.7, f'(n={cm[i,j]})',
                       ha='center', va='center', fontsize=8, color='gray')

        ax.set_title(f"{exp_name.replace('Multi-Project', 'MP')}\n"
                    f"Features: {exp_data['state_dim']} state + {exp_data['action_dim']} action",
                    fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    print(f"保存: {output_dir / 'confusion_matrices.png'}")

def analyze_class_imbalance_impact(results: Dict, output_dir: Path):
    """クラス不均衡の影響分析"""
    print("\n" + "="*100)
    print("クラス不均衡の影響分析")
    print("="*100)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Class Imbalance Impact Analysis', fontsize=16, fontweight='bold')

    data_for_plot = []

    for exp_name, exp_data in results.items():
        print(f"\n【{exp_name}】")
        print("-"*100)

        if 'predictions_df' not in exp_data:
            continue

        # 訓練期間ごとのクラス分布を確認
        for metrics in exp_data['metrics']:
            train_period = metrics['train_period']
            eval_period = metrics['eval_period']

            positive_count = metrics.get('positive_count', 0)
            negative_count = metrics.get('negative_count', 0)
            total = positive_count + negative_count

            if total > 0:
                positive_rate = positive_count / total

                print(f"{train_period} → {eval_period}:")
                print(f"  Positive: {positive_count} ({positive_rate:.1%})")
                print(f"  Negative: {negative_count} ({1-positive_rate:.1%})")
                print(f"  F1: {metrics.get('f1_score', 0):.4f}")
                print(f"  Precision: {metrics.get('precision', 0):.4f}")
                print(f"  Recall: {metrics.get('recall', 0):.4f}")

                data_for_plot.append({
                    'experiment': exp_name.replace('Multi-Project', 'MP'),
                    'positive_rate': positive_rate,
                    'f1_score': metrics.get('f1_score', 0),
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0),
                    'auc_roc': metrics.get('auc_roc', 0)
                })

    # プロット
    df_plot = pd.DataFrame(data_for_plot)

    metrics_to_plot = ['f1_score', 'precision', 'recall', 'auc_roc']
    metric_names = ['F1 Score', 'Precision', 'Recall', 'AUC-ROC']

    for idx, (metric, metric_name) in enumerate(zip(metrics_to_plot, metric_names)):
        ax = axes[idx // 2, idx % 2]

        for exp in df_plot['experiment'].unique():
            subset = df_plot[df_plot['experiment'] == exp]
            ax.scatter(subset['positive_rate'], subset[metric],
                      label=exp, alpha=0.6, s=100)

        ax.set_xlabel('Positive Rate (Acceptance Rate)', fontweight='bold')
        ax.set_ylabel(metric_name, fontweight='bold')
        ax.set_title(f'{metric_name} vs Class Imbalance', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'class_imbalance_impact.png', dpi=300, bbox_inches='tight')
    print(f"\n保存: {output_dir / 'class_imbalance_impact.png'}")

def main():
    results_dir = Path("/Users/kazuki-h/research/multiproject_research/results")
    output_dir = Path("/Users/kazuki-h/research/multiproject_research/docs/figures/comprehensive")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("結果を読み込み中...")
    results = load_all_results(results_dir)

    print(f"\n読み込まれた実験: {list(results.keys())}")
    for exp_name, exp_data in results.items():
        print(f"\n{exp_name}:")
        print(f"  State次元: {exp_data['state_dim']}")
        print(f"  Action次元: {exp_data['action_dim']}")
        print(f"  特徴量: {exp_data['features']}")
        print(f"  メトリクス数: {len(exp_data['metrics'])}")
        if 'predictions_df' in exp_data:
            print(f"  予測数: {len(exp_data['predictions_df'])}")

    print("\n" + "="*100)
    print("可視化を生成中...")
    print("="*100)

    print("\n1. 特徴量次元の比較...")
    plot_feature_comparison(results, output_dir)

    print("\n2. 特徴量次元と性能の関係...")
    plot_performance_vs_features(results, output_dir)

    print("\n3. 予測確率の分布...")
    plot_prediction_distribution(results, output_dir)

    print("\n4. 混同行列...")
    plot_confusion_matrices(results, output_dir)

    print("\n5. クラス不均衡の影響分析...")
    analyze_class_imbalance_impact(results, output_dir)

    print("\n6. 統計的比較...")
    statistical_comparison(results, output_dir)

    print(f"\n\n全ての分析が完了しました: {output_dir}")

if __name__ == "__main__":
    main()

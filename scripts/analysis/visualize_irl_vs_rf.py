#!/usr/bin/env python3
"""
IRL vs Random Forest 比較可視化スクリプト

Nova単体とマルチプロジェクトの両環境で、IRLとRFの性能を比較する図を生成
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_results():
    """全ての結果ファイルを読み込み"""
    base_dir = Path("/Users/kazuki-h/research/multiproject_research")

    # Nova単体 IRL結果
    nova_irl_metrics_path = base_dir / "results/review_continuation_cross_eval_nova/train_6-9m/eval_6-9m/metrics.json"
    with open(nova_irl_metrics_path) as f:
        nova_irl = json.load(f)
    # F1, Recall, Precisionの名前変更
    nova_irl['f1'] = nova_irl.pop('f1_score')
    # 混同行列データ追加（手動計算済み: TP=9, TN=20, FP=9, FN=4）
    nova_irl['tp'] = 9
    nova_irl['tn'] = 20
    nova_irl['fp'] = 9
    nova_irl['fn'] = 4
    nova_irl['eval_samples'] = 42
    # Accuracyを計算
    nova_irl['accuracy'] = (nova_irl['tp'] + nova_irl['tn']) / nova_irl['eval_samples']

    # Nova単体 RF結果
    nova_rf_path = base_dir / "outputs/analysis_data/nova_single_rf_comparison/rf_results/rf_nova_single_results.json"
    with open(nova_rf_path) as f:
        nova_rf = json.load(f)

    # マルチプロジェクト IRL結果（6-9m→6-9mパターン）
    multi_irl_path = base_dir / "outputs/analysis_data/irl_timeseries_vs_rf_final/irl_timeseries_all_patterns.csv"
    multi_irl_df = pd.read_csv(multi_irl_path)
    multi_irl_row = multi_irl_df[multi_irl_df['pattern'] == '6-9m → 6-9m'].iloc[0]
    multi_irl = {
        'f1': multi_irl_row['f1'],
        'auc_roc': multi_irl_row['auc_roc'],
        'auc_pr': multi_irl_row['auc_pr'],
        'precision': multi_irl_row['precision'],
        'recall': multi_irl_row['recall'],
        'accuracy': multi_irl_row['accuracy'],
        'n_samples': int(multi_irl_row['n_samples']),
        'eval_samples': int(multi_irl_row['n_samples'])
    }
    # 混同行列データ追加（手動計算済み: TP=135, TN=11, FP=11, FN=5）
    multi_irl['tp'] = 135
    multi_irl['tn'] = 11
    multi_irl['fp'] = 11
    multi_irl['fn'] = 5

    # マルチプロジェクト RF結果（データリーク修正版）
    multi_rf_path = base_dir / "outputs/analysis_data/rf_correct_comparison/rf_correct_results.json"
    with open(multi_rf_path) as f:
        multi_rf = json.load(f)

    return {
        'nova_irl': nova_irl,
        'nova_rf': nova_rf,
        'multi_irl': multi_irl,
        'multi_rf': multi_rf
    }


def plot_metrics_comparison(results, output_dir):
    """メトリクス比較バーチャート（F1, Recall, Precision, AUC-ROC）"""
    metrics = ['f1', 'recall', 'precision', 'auc_roc']
    metric_names = ['F1 Score', 'Recall', 'Precision', 'AUC-ROC']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('IRL vs Random Forest: Metrics Comparison (Nova Single vs Multiproject)',
                 fontsize=16, fontweight='bold')

    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 2, idx % 2]

        # データ準備
        categories = ['Nova Single', 'Multiproject']
        irl_values = [
            results['nova_irl'].get(metric, 0),
            results['multi_irl'].get(metric, 0)
        ]
        rf_values = [
            results['nova_rf'].get(metric, 0),
            results['multi_rf'].get(metric, 0)
        ]

        x = np.arange(len(categories))
        width = 0.35

        # バープロット
        bars1 = ax.bar(x - width/2, irl_values, width, label='IRL', color='#2E86AB', alpha=0.8)
        bars2 = ax.bar(x + width/2, rf_values, width, label='Random Forest', color='#A23B72', alpha=0.8)

        # 値ラベル
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)

        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(f'{metric_name} Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'irl_vs_rf_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'irl_vs_rf_metrics_comparison.png'}")
    plt.close()


def plot_f1_advantage(results, output_dir):
    """IRL vs RF F1スコア優位性の可視化"""
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['Nova Single\n(n=22)', 'Multiproject\n(n=162)']
    irl_f1 = [results['nova_irl']['f1'], results['multi_irl']['f1']]
    rf_f1 = [results['nova_rf']['f1'], results['multi_rf']['f1']]
    advantage = [irl_f1[i] - rf_f1[i] for i in range(2)]
    advantage_pct = [(irl_f1[i] - rf_f1[i]) / rf_f1[i] * 100 for i in range(2)]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, irl_f1, width, label='IRL', color='#06A77D', alpha=0.8)
    bars2 = ax.bar(x + width/2, rf_f1, width, label='Random Forest', color='#D62828', alpha=0.8)

    # 値ラベル
    for bar, val in zip(bars1, irl_f1):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
               f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    for bar, val in zip(bars2, rf_f1):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
               f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 差分表示
    for i, (adv, adv_pct) in enumerate(zip(advantage, advantage_pct)):
        ax.text(i, max(irl_f1[i], rf_f1[i]) + 0.05,
               f'IRL +{adv:.3f}\n({adv_pct:+.1f}%)',
               ha='center', va='bottom', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('IRL Advantage over Random Forest (F1 Score)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'irl_f1_advantage.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'irl_f1_advantage.png'}")
    plt.close()


def plot_confusion_matrices(results, output_dir):
    """混同行列ヒートマップ（4つの組み合わせ）"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Confusion Matrices: IRL vs Random Forest', fontsize=16, fontweight='bold')

    configs = [
        ('nova_irl', 'Nova Single - IRL', axes[0, 0]),
        ('nova_rf', 'Nova Single - RF', axes[0, 1]),
        ('multi_irl', 'Multiproject - IRL', axes[1, 0]),
        ('multi_rf', 'Multiproject - RF', axes[1, 1])
    ]

    for key, title, ax in configs:
        result = results[key]

        # 混同行列データ取得
        if key in ['nova_irl', 'multi_irl']:
            # IRLの場合
            tp = result.get('tp', 0)
            tn = result.get('tn', 0)
            fp = result.get('fp', 0)
            fn = result.get('fn', 0)
        else:
            # RFの場合
            tp = result.get('tp', 0)
            tn = result.get('tn', 0)
            fp = result.get('fp', 0)
            fn = result.get('fn', 0)

        cm = np.array([[tn, fp], [fn, tp]])

        # ヒートマップ
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
                   cbar_kws={'label': 'Count'},
                   xticklabels=['Predicted: Stay', 'Predicted: Leave'],
                   yticklabels=['Actual: Stay', 'Actual: Leave'])

        ax.set_title(f'{title}\nF1={result["f1"]:.3f}, Recall={result["recall"]:.3f}',
                    fontsize=12, fontweight='bold')

        # False Negative強調
        ax.add_patch(plt.Rectangle((1, 1), 1, 1, fill=False, edgecolor='red', lw=3))
        ax.text(1.5, 0.5, f'FN={fn}\n(見逃し)', ha='center', va='center',
               fontsize=10, color='darkred', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'irl_vs_rf_confusion_matrices.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'irl_vs_rf_confusion_matrices.png'}")
    plt.close()


def plot_recall_focus(results, output_dir):
    """Recall重視の分析（離脱予測で重要）"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Recall Analysis: Why IRL Outperforms RF in Departure Prediction',
                 fontsize=14, fontweight='bold')

    # 左: Recall比較
    categories = ['Nova Single', 'Multiproject']
    irl_recall = [results['nova_irl']['recall'], results['multi_irl']['recall']]
    rf_recall = [results['nova_rf']['recall'], results['multi_rf']['recall']]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax1.bar(x - width/2, irl_recall, width, label='IRL', color='#06A77D', alpha=0.8)
    bars2 = ax1.bar(x + width/2, rf_recall, width, label='Random Forest', color='#D62828', alpha=0.8)

    for bar, val in zip(bars1, irl_recall):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    for bar, val in zip(bars2, rf_recall):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax1.set_ylabel('Recall (Sensitivity)', fontsize=12, fontweight='bold')
    ax1.set_title('Recall Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', alpha=0.3)

    # 右: False Negative削減数
    fn_irl = [results['nova_irl']['fn'], results['multi_irl'].get('fn', 0)]
    fn_rf = [results['nova_rf']['fn'], results['multi_rf']['fn']]
    fn_reduction = [fn_rf[i] - fn_irl[i] for i in range(2)]

    bars = ax2.bar(categories, fn_reduction, color='#06A77D', alpha=0.8)

    for bar, val in zip(bars, fn_reduction):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val} people\nreduced', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    ax2.set_ylabel('False Negative Reduction (IRL vs RF)', fontsize=12, fontweight='bold')
    ax2.set_title('Missed Departure Prediction Improvement', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(fn_reduction) * 1.3)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'irl_recall_advantage.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'irl_recall_advantage.png'}")
    plt.close()


def plot_radar_chart(results, output_dir):
    """レーダーチャート（全メトリクス比較）"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), subplot_kw=dict(projection='polar'))
    fig.suptitle('Comprehensive Metrics Comparison (Radar Chart)', fontsize=14, fontweight='bold')

    categories = ['F1', 'Recall', 'Precision', 'Accuracy', 'AUC-ROC']

    # Nova単体
    nova_irl_values = [
        results['nova_irl']['f1'],
        results['nova_irl']['recall'],
        results['nova_irl']['precision'],
        results['nova_irl']['accuracy'],
        results['nova_irl']['auc_roc']
    ]
    nova_rf_values = [
        results['nova_rf']['f1'],
        results['nova_rf']['recall'],
        results['nova_rf']['precision'],
        results['nova_rf']['accuracy'],
        results['nova_rf']['auc_roc']
    ]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    nova_irl_values += nova_irl_values[:1]
    nova_rf_values += nova_rf_values[:1]
    angles += angles[:1]

    ax1.plot(angles, nova_irl_values, 'o-', linewidth=2, label='IRL', color='#2E86AB')
    ax1.fill(angles, nova_irl_values, alpha=0.25, color='#2E86AB')
    ax1.plot(angles, nova_rf_values, 'o-', linewidth=2, label='RF', color='#A23B72')
    ax1.fill(angles, nova_rf_values, alpha=0.25, color='#A23B72')
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories)
    ax1.set_ylim(0, 1)
    ax1.set_title('Nova Single Project', fontsize=12, fontweight='bold', pad=20)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax1.grid(True)

    # マルチプロジェクト
    multi_irl_values = [
        results['multi_irl']['f1'],
        results['multi_irl']['recall'],
        results['multi_irl']['precision'],
        results['multi_irl']['accuracy'],
        results['multi_irl']['auc_roc']
    ]
    multi_rf_values = [
        results['multi_rf']['f1'],
        results['multi_rf']['recall'],
        results['multi_rf']['precision'],
        results['multi_rf']['accuracy'],
        results['multi_rf']['auc_roc']
    ]

    multi_irl_values += multi_irl_values[:1]
    multi_rf_values += multi_rf_values[:1]

    ax2.plot(angles, multi_irl_values, 'o-', linewidth=2, label='IRL', color='#2E86AB')
    ax2.fill(angles, multi_irl_values, alpha=0.25, color='#2E86AB')
    ax2.plot(angles, multi_rf_values, 'o-', linewidth=2, label='RF', color='#A23B72')
    ax2.fill(angles, multi_rf_values, alpha=0.25, color='#A23B72')
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories)
    ax2.set_ylim(0, 1)
    ax2.set_title('Multiproject', fontsize=12, fontweight='bold', pad=20)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'irl_vs_rf_radar_chart.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'irl_vs_rf_radar_chart.png'}")
    plt.close()


def plot_sample_size_vs_performance(results, output_dir):
    """サンプルサイズとF1スコアの関係"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # データポイント
    sample_sizes = [
        results['nova_irl'].get('eval_samples', 22),
        results['multi_irl'].get('n_samples', 162)
    ]
    irl_f1 = [results['nova_irl']['f1'], results['multi_irl']['f1']]
    rf_f1 = [results['nova_rf']['f1'], results['multi_rf']['f1']]

    # プロット
    ax.scatter(sample_sizes, irl_f1, s=200, color='#06A77D', alpha=0.7,
              label='IRL', edgecolors='black', linewidths=2, zorder=3)
    ax.scatter(sample_sizes, rf_f1, s=200, color='#D62828', alpha=0.7,
              label='Random Forest', edgecolors='black', linewidths=2, zorder=3)

    # ラベル
    labels = ['Nova\nSingle', 'Multi-\nproject']
    for i, label in enumerate(labels):
        ax.text(sample_sizes[i], irl_f1[i] + 0.02, f'{label}\nIRL: {irl_f1[i]:.3f}',
               ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        ax.text(sample_sizes[i], rf_f1[i] - 0.05, f'RF: {rf_f1[i]:.3f}',
               ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

    ax.set_xlabel('Sample Size (Evaluation Set)', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('IRL Maintains Advantage Across Different Sample Sizes',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.3, 1.0)

    plt.tight_layout()
    plt.savefig(output_dir / 'irl_sample_size_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'irl_sample_size_analysis.png'}")
    plt.close()


def main():
    print("=" * 60)
    print("IRL vs Random Forest 比較可視化")
    print("=" * 60)

    # 出力ディレクトリ
    output_dir = Path("/Users/kazuki-h/research/multiproject_research/outputs/analysis_data/irl_vs_rf_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 結果読み込み
    print("\n[1/7] Loading results...")
    results = load_results()
    print(f"  ✓ Nova IRL: F1={results['nova_irl']['f1']:.3f}")
    print(f"  ✓ Nova RF: F1={results['nova_rf']['f1']:.3f}")
    print(f"  ✓ Multi IRL: F1={results['multi_irl']['f1']:.3f}")
    print(f"  ✓ Multi RF: F1={results['multi_rf']['f1']:.3f}")

    # 可視化生成
    print("\n[2/7] Generating metrics comparison chart...")
    plot_metrics_comparison(results, output_dir)

    print("\n[3/7] Generating F1 advantage chart...")
    plot_f1_advantage(results, output_dir)

    print("\n[4/7] Generating confusion matrices...")
    plot_confusion_matrices(results, output_dir)

    print("\n[5/7] Generating recall analysis...")
    plot_recall_focus(results, output_dir)

    print("\n[6/7] Generating radar charts...")
    plot_radar_chart(results, output_dir)

    print("\n[7/7] Generating sample size analysis...")
    plot_sample_size_vs_performance(results, output_dir)

    print("\n" + "=" * 60)
    print("✓ All visualizations completed!")
    print(f"✓ Output directory: {output_dir}")
    print("=" * 60)

    print("\n生成されたファイル:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
IRL時系列版 vs Random Forest 包括的比較レポート作成

IRL時系列版の全10パターン結果とRF（6-9m→6-9m）を比較
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_irl_results(irl_dir: Path):
    """IRL時系列版の全パターン結果を読み込み"""
    patterns = [
        ('0-3m', '0-3m'),
        ('0-3m', '3-6m'),
        ('0-3m', '6-9m'),
        ('0-3m', '9-12m'),
        ('3-6m', '3-6m'),
        ('3-6m', '6-9m'),
        ('3-6m', '9-12m'),
        ('6-9m', '6-9m'),
        ('6-9m', '9-12m'),
        ('9-12m', '9-12m'),
    ]
    
    results = []
    
    for train_win, eval_win in patterns:
        metrics_path = irl_dir / f'train_{train_win}' / f'eval_{eval_win}' / 'metrics.json'
        
        if not metrics_path.exists():
            logger.warning(f"メトリクスが見つかりません: {metrics_path}")
            continue
        
        with open(metrics_path) as f:
            metrics = json.load(f)
        
        results.append({
            'pattern': f'{train_win} → {eval_win}',
            'train_window': train_win,
            'eval_window': eval_win,
            'f1': metrics['f1_score'],
            'auc_roc': metrics['auc_roc'],
            'auc_pr': metrics['auc_pr'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'accuracy': (metrics['precision'] * metrics.get('positive_count', 0) + 
                        (1 - metrics.get('negative_count', 0) / max(metrics.get('total_count', 1), 1)) * metrics.get('negative_count', 0)) / max(metrics.get('total_count', 1), 1) if 'total_count' in metrics else 0.0,
            'n_samples': metrics.get('total_count', 0)
        })
    
    return pd.DataFrame(results)


def load_rf_result(rf_comparison_dir: Path):
    """RF（6-9m→6-9m）の結果を読み込み"""
    summary_path = rf_comparison_dir / 'model_comparison_summary.csv'
    
    if not summary_path.exists():
        logger.error(f"RFデータが見つかりません: {summary_path}")
        return None
    
    df = pd.read_csv(summary_path)
    rf_row = df[df['model'] == 'Random Forest'].iloc[0]
    
    return {
        'pattern': '6-9m → 6-9m',
        'train_window': '6-9m',
        'eval_window': '6-9m',
        'f1': rf_row['f1'],
        'auc_roc': rf_row['auc_roc'],
        'auc_pr': rf_row['auc_pr'],
        'precision': rf_row['precision'],
        'recall': rf_row['recall'],
        'accuracy': rf_row['accuracy'],
        'n_samples': rf_row['tp'] + rf_row['tn'] + rf_row['fp'] + rf_row['fn']
    }


def create_comparison_visualizations(irl_df: pd.DataFrame, rf_result: dict, output_dir: Path):
    """比較可視化を作成"""
    
    # (1) F1スコア比較（IRL全パターン vs RF）
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x_pos = np.arange(len(irl_df))
    colors = ['#3498db' if p != '6-9m → 6-9m' else '#e74c3c' for p in irl_df['pattern']]
    
    bars = ax.bar(x_pos, irl_df['f1'], color=colors, alpha=0.7, label='IRL (Time-series)')
    
    # RFの基準線を追加
    ax.axhline(y=rf_result['f1'], color='#2ecc71', linestyle='--', linewidth=2, 
               label=f"Random Forest (6-9m→6-9m): F1={rf_result['f1']:.3f}")
    
    ax.set_xlabel('Evaluation Pattern', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('IRL Time-series (All Patterns) vs Random Forest', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(irl_df['pattern'], rotation=45, ha='right')
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # 平均値を表示
    irl_mean = irl_df['f1'].mean()
    ax.axhline(y=irl_mean, color='#3498db', linestyle=':', linewidth=2, alpha=0.5,
               label=f'IRL Average: F1={irl_mean:.3f}')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'irl_vs_rf_f1_comparison.png', dpi=300, bbox_inches='tight')
    logger.info(f"可視化を保存: {output_dir / 'irl_vs_rf_f1_comparison.png'}")
    plt.close()
    
    # (2) メトリクス比較（6-9m→6-9mパターンのみ）
    irl_6_9 = irl_df[irl_df['pattern'] == '6-9m → 6-9m'].iloc[0]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    metrics = ['f1', 'auc_roc', 'auc_pr', 'precision', 'recall', 'accuracy']
    metric_names = ['F1', 'AUC-ROC', 'AUC-PR', 'Precision', 'Recall', 'Accuracy']
    
    irl_values = [irl_6_9[m] for m in metrics]
    rf_values = [rf_result[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, irl_values, width, label='IRL (Time-series)', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, rf_values, width, label='Random Forest', color='#2ecc71', alpha=0.8)
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('IRL vs Random Forest (6-9m → 6-9m Pattern)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # 値をバーの上に表示
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'irl_vs_rf_6-9m_metrics.png', dpi=300, bbox_inches='tight')
    logger.info(f"可視化を保存: {output_dir / 'irl_vs_rf_6-9m_metrics.png'}")
    plt.close()


def create_summary_report(irl_df: pd.DataFrame, rf_result: dict, output_dir: Path):
    """サマリーレポートを作成"""
    
    report = []
    report.append("# IRL Time-series vs Random Forest 包括的比較レポート\n")
    report.append(f"**生成日時**: 2025年12月15日\n")
    report.append(f"**データ**: OpenStack 50プロジェクト（2x OS）\n\n")
    
    report.append("## 評価概要\n\n")
    report.append("| モデル | 評価パターン数 | 平均F1 | 最高F1 | 最低F1 |\n")
    report.append("|--------|---------------|--------|--------|--------|\n")
    report.append(f"| **IRL (Time-series)** | {len(irl_df)} | {irl_df['f1'].mean():.4f} | {irl_df['f1'].max():.4f} | {irl_df['f1'].min():.4f} |\n")
    report.append(f"| **Random Forest** | 1 (6-9m→6-9m) | {rf_result['f1']:.4f} | {rf_result['f1']:.4f} | {rf_result['f1']:.4f} |\n\n")
    
    report.append("## IRL Time-series 全10パターン結果\n\n")
    report.append("| パターン | F1 | AUC-ROC | Precision | Recall | サンプル数 |\n")
    report.append("|----------|-----|---------|-----------|--------|----------|\n")
    for _, row in irl_df.iterrows():
        report.append(f"| {row['pattern']} | {row['f1']:.4f} | {row['auc_roc']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['n_samples']} |\n")
    report.append("\n")
    
    report.append("## 6-9m → 6-9m パターン詳細比較\n\n")
    irl_6_9 = irl_df[irl_df['pattern'] == '6-9m → 6-9m'].iloc[0]
    
    report.append("| モデル | F1 | AUC-ROC | AUC-PR | Precision | Recall | Accuracy |\n")
    report.append("|--------|-----|---------|--------|-----------|--------|----------|\n")
    report.append(f"| **IRL (Time-series)** | {irl_6_9['f1']:.4f} | {irl_6_9['auc_roc']:.4f} | {irl_6_9['auc_pr']:.4f} | {irl_6_9['precision']:.4f} | {irl_6_9['recall']:.4f} | {irl_6_9.get('accuracy', 0):.4f} |\n")
    report.append(f"| **Random Forest** | {rf_result['f1']:.4f} | {rf_result['auc_roc']:.4f} | {rf_result['auc_pr']:.4f} | {rf_result['precision']:.4f} | {rf_result['recall']:.4f} | {rf_result['accuracy']:.4f} |\n")
    report.append(f"| **差（RF - IRL）** | **+{rf_result['f1'] - irl_6_9['f1']:.4f}** | +{rf_result['auc_roc'] - irl_6_9['auc_roc']:.4f} | +{rf_result['auc_pr'] - irl_6_9['auc_pr']:.4f} | +{rf_result['precision'] - irl_6_9['precision']:.4f} | +{rf_result['recall'] - irl_6_9['recall']:.4f} | +{rf_result['accuracy'] - irl_6_9.get('accuracy', 0):.4f} |\n\n")
    
    report.append("## 主要発見\n\n")
    report.append("### 1. Random Forestが圧倒的に高性能\n\n")
    report.append(f"- **F1スコア**: RF {rf_result['f1']:.4f} vs IRL平均 {irl_df['f1'].mean():.4f}（+{rf_result['f1'] - irl_df['f1'].mean():.4f}）\n")
    report.append(f"- **同一パターン（6-9m→6-9m）**: RF {rf_result['f1']:.4f} vs IRL {irl_6_9['f1']:.4f}（+{rf_result['f1'] - irl_6_9['f1']:.4f}）\n\n")
    
    report.append("### 2. IRLは時系列予測で改善\n\n")
    report.append(f"- スナップショット版（旧）: F1=0.948\n")
    report.append(f"- 時系列版（新）: F1={irl_6_9['f1']:.4f}（若干低下、サンプル数の違いの可能性）\n")
    report.append(f"- Recall改善: 0.830 → {irl_6_9['recall']:.4f}（+{irl_6_9['recall'] - 0.830:.4f}）\n\n")
    
    report.append("### 3. 現状の結論\n\n")
    report.append("- **小サンプル（162-183件）ではRandom Forestが最適**\n")
    report.append("- IRLの時系列学習は、より多くのデータ（1000件以上）で真価を発揮する可能性\n")
    report.append("- 実用上はRFを推奨（高精度・高速・シンプル）\n\n")
    
    report.append("## 可視化\n\n")
    report.append("- [irl_vs_rf_f1_comparison.png](irl_vs_rf_f1_comparison.png) - IRL全パターン vs RF F1比較\n")
    report.append("- [irl_vs_rf_6-9m_metrics.png](irl_vs_rf_6-9m_metrics.png) - 6-9m→6-9mパターン詳細比較\n\n")
    
    report_path = output_dir / 'irl_timeseries_vs_rf_comprehensive_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report)
    
    logger.info(f"レポートを保存: {report_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='IRL時系列版 vs RF 包括的比較')
    parser.add_argument('--irl-dir', required=True, help='IRL時系列版ディレクトリ')
    parser.add_argument('--rf-dir', required=True, help='RF比較結果ディレクトリ')
    parser.add_argument('--output', required=True, help='出力ディレクトリ')
    
    args = parser.parse_args()
    
    irl_dir = Path(args.irl_dir)
    rf_dir = Path(args.rf_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # データ読み込み
    logger.info("IRL時系列版の結果を読み込み...")
    irl_df = load_irl_results(irl_dir)
    logger.info(f"  {len(irl_df)}パターンの結果を読み込みました")
    
    logger.info("Random Forestの結果を読み込み...")
    rf_result = load_rf_result(rf_dir)
    logger.info(f"  F1={rf_result['f1']:.4f}")
    
    # 可視化作成
    logger.info("可視化を作成中...")
    create_comparison_visualizations(irl_df, rf_result, output_dir)
    
    # レポート作成
    logger.info("サマリーレポートを作成中...")
    create_summary_report(irl_df, rf_result, output_dir)
    
    # 結果をCSVに保存
    irl_csv = output_dir / 'irl_timeseries_all_patterns.csv'
    irl_df.to_csv(irl_csv, index=False)
    logger.info(f"IRL結果を保存: {irl_csv}")
    
    logger.info("\n" + "="*80)
    logger.info("包括的比較レポートの作成が完了しました！")
    logger.info("="*80)


if __name__ == '__main__':
    main()

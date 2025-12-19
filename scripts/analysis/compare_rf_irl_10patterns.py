#!/usr/bin/env python3
"""
RF vs IRL 10パターン比較分析

IRL結果（16パターン）から10パターンを抽出し、RF結果と比較する。
"""

import json
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_irl_results():
    """IRLの結果を読み込み（16パターン）"""
    irl_dir = Path("/Users/kazuki-h/research/multiproject_research/results/review_continuation_cross_eval_nova")

    # 各時間窓の結果を読み込み
    train_periods = ['0-3m', '3-6m', '6-9m', '9-12m']
    eval_periods = ['0-3m', '3-6m', '6-9m', '9-12m']

    results = []

    for train_p in train_periods:
        train_dir = irl_dir / f"train_{train_p}"

        if not train_dir.exists():
            logger.warning(f"ディレクトリが見つかりません: {train_dir}")
            continue

        for eval_p in eval_periods:
            eval_dir = train_dir / f"eval_{eval_p}"
            metrics_file = eval_dir / "metrics.json"

            if not metrics_file.exists():
                logger.warning(f"メトリクスファイルが見つかりません: {metrics_file}")
                continue

            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            result = {
                'pattern': f"{train_p} → {eval_p}",
                'train_period': train_p,
                'eval_period': eval_p,
                'f1': metrics.get('F1', 0),
                'auc_roc': metrics.get('AUC_ROC', 0),
                'auc_pr': metrics.get('AUC_PR', 0),
                'precision': metrics.get('PRECISION', 0),
                'recall': metrics.get('RECALL', 0),
                'model': 'IRL'
            }
            results.append(result)

    logger.info(f"IRL結果読み込み完了: {len(results)} パターン")
    return results


def load_rf_results():
    """RFの結果を読み込み（10パターン）"""
    rf_file = Path("/Users/kazuki-h/research/multiproject_research/outputs/rf_nova_10patterns_irl_aligned_v2/all_results.json")

    if not rf_file.exists():
        raise FileNotFoundError(f"RFファイルが見つかりません: {rf_file}")

    with open(rf_file, 'r') as f:
        rf_results = json.load(f)

    results = []
    for r in rf_results:
        # パターン名から期間を抽出（例: "0-3m → 3-6m"）
        pattern = r['pattern']
        parts = pattern.split(' → ')

        result = {
            'pattern': pattern,
            'train_period': parts[0],
            'eval_period': parts[1],
            'f1': r['f1'],
            'auc_roc': r['auc_roc'],
            'auc_pr': r['auc_pr'],
            'precision': r['precision'],
            'recall': r['recall'],
            'model': 'RF'
        }
        results.append(result)

    logger.info(f"RF結果読み込み完了: {len(results)} パターン")
    return results


def extract_10_patterns(irl_results):
    """IRLの16パターンから10パターンを抽出（上三角行列）"""
    # 10パターンの定義（過去→未来 or 同期間のみ）
    target_patterns = [
        "0-3m → 0-3m",
        "0-3m → 3-6m",
        "0-3m → 6-9m",
        "0-3m → 9-12m",
        "3-6m → 3-6m",
        "3-6m → 6-9m",
        "3-6m → 9-12m",
        "6-9m → 6-9m",
        "6-9m → 9-12m",
        "9-12m → 9-12m",
    ]

    filtered = [r for r in irl_results if r['pattern'] in target_patterns]
    logger.info(f"IRL 10パターン抽出完了: {len(filtered)} パターン")

    return filtered


def compare_results(rf_results, irl_10_results):
    """RFとIRLの結果を比較"""
    # データフレームに変換
    df_rf = pd.DataFrame(rf_results)
    df_irl = pd.DataFrame(irl_10_results)

    # マージ
    df_merged = pd.merge(
        df_rf[['pattern', 'f1', 'auc_roc', 'precision', 'recall']],
        df_irl[['pattern', 'f1', 'auc_roc', 'precision', 'recall']],
        on='pattern',
        suffixes=('_RF', '_IRL')
    )

    logger.info("\n" + "=" * 100)
    logger.info("RF vs IRL 比較（10パターン）")
    logger.info("=" * 100)

    # メトリクスごとの比較
    metrics = ['f1', 'auc_roc', 'precision', 'recall']

    for metric in metrics:
        rf_col = f"{metric}_RF"
        irl_col = f"{metric}_IRL"

        rf_mean = df_merged[rf_col].mean()
        irl_mean = df_merged[irl_col].mean()
        diff = irl_mean - rf_mean
        diff_pct = (diff / rf_mean * 100) if rf_mean > 0 else 0

        logger.info(f"\n{metric.upper()}:")
        logger.info(f"  RF平均:   {rf_mean:.4f}")
        logger.info(f"  IRL平均:  {irl_mean:.4f}")
        logger.info(f"  差分:     {diff:+.4f} ({diff_pct:+.2f}%)")

        # 勝敗カウント
        wins = (df_merged[irl_col] > df_merged[rf_col]).sum()
        ties = (df_merged[irl_col] == df_merged[rf_col]).sum()
        losses = (df_merged[irl_col] < df_merged[rf_col]).sum()
        logger.info(f"  勝敗:     IRL勝ち={wins}, 引き分け={ties}, RF勝ち={losses}")

    logger.info("=" * 100)

    return df_merged


def create_comparison_visualizations(df_merged, output_dir):
    """比較可視化を作成"""
    metrics = ['f1', 'auc_roc', 'precision', 'recall']
    metric_labels = {
        'f1': 'F1 Score',
        'auc_roc': 'AUC-ROC',
        'precision': 'Precision',
        'recall': 'Recall'
    }

    # 1. 各メトリクスのバープロット比較
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]

        x = range(len(df_merged))
        width = 0.35

        rf_values = df_merged[f"{metric}_RF"]
        irl_values = df_merged[f"{metric}_IRL"]

        ax.bar([p - width/2 for p in x], rf_values, width, label='RF', alpha=0.8)
        ax.bar([p + width/2 for p in x], irl_values, width, label='IRL', alpha=0.8)

        ax.set_xlabel('Pattern', fontsize=10)
        ax.set_ylabel(metric_labels[metric], fontsize=10)
        ax.set_title(f'{metric_labels[metric]} Comparison (10 Patterns)', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df_merged['pattern'], rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "comparison_barplots.png", dpi=150, bbox_inches='tight')
    logger.info(f"✓ バープロット保存: {output_dir / 'comparison_barplots.png'}")
    plt.close()

    # 2. 散布図（RF vs IRL）
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]

        rf_values = df_merged[f"{metric}_RF"]
        irl_values = df_merged[f"{metric}_IRL"]

        ax.scatter(rf_values, irl_values, s=100, alpha=0.6)

        # 対角線（完全一致）
        min_val = min(rf_values.min(), irl_values.min())
        max_val = max(rf_values.max(), irl_values.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect Match')

        # パターンラベル
        for j, row in df_merged.iterrows():
            ax.annotate(row['pattern'], (row[f"{metric}_RF"], row[f"{metric}_IRL"]),
                       fontsize=6, alpha=0.7)

        ax.set_xlabel(f'RF {metric_labels[metric]}', fontsize=10)
        ax.set_ylabel(f'IRL {metric_labels[metric]}', fontsize=10)
        ax.set_title(f'{metric_labels[metric]}: RF vs IRL', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "comparison_scatterplots.png", dpi=150, bbox_inches='tight')
    logger.info(f"✓ 散布図保存: {output_dir / 'comparison_scatterplots.png'}")
    plt.close()

    # 3. 差分ヒートマップ（各パターンでの差分）
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # 差分計算（IRL - RF）
        diff = df_merged[f"{metric}_IRL"] - df_merged[f"{metric}_RF"]

        # ヒートマップ用にreshape（10パターンを並べる）
        diff_matrix = diff.values.reshape(1, -1)

        sns.heatmap(diff_matrix, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                   xticklabels=df_merged['pattern'], yticklabels=['Diff'],
                   ax=ax, cbar_kws={'label': 'IRL - RF'})

        ax.set_title(f'{metric_labels[metric]} Difference (IRL - RF)', fontsize=12, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "comparison_diff_heatmap.png", dpi=150, bbox_inches='tight')
    logger.info(f"✓ 差分ヒートマップ保存: {output_dir / 'comparison_diff_heatmap.png'}")
    plt.close()


def save_comparison_report(df_merged, output_dir):
    """比較レポートを保存"""
    report_path = output_dir / "RF_vs_IRL_10patterns_comparison.md"

    with open(report_path, 'w') as f:
        f.write("# RF vs IRL 10パターン比較分析レポート\n\n")
        f.write(f"生成日時: {pd.Timestamp.now()}\n\n")

        f.write("## 概要\n\n")
        f.write("- **RF**: Random Forest (14次元特徴量)\n")
        f.write("- **IRL**: Inverse Reinforcement Learning\n")
        f.write("- **評価パターン**: 10パターン（過去→未来 or 同期間のみ）\n\n")

        f.write("## メトリクス比較サマリー\n\n")

        metrics = ['f1', 'auc_roc', 'precision', 'recall']

        summary_data = []
        for metric in metrics:
            rf_col = f"{metric}_RF"
            irl_col = f"{metric}_IRL"

            rf_mean = df_merged[rf_col].mean()
            irl_mean = df_merged[irl_col].mean()
            diff = irl_mean - rf_mean
            diff_pct = (diff / rf_mean * 100) if rf_mean > 0 else 0

            wins = (df_merged[irl_col] > df_merged[rf_col]).sum()
            ties = (df_merged[irl_col] == df_merged[rf_col]).sum()
            losses = (df_merged[irl_col] < df_merged[rf_col]).sum()

            summary_data.append({
                'Metric': metric.upper(),
                'RF平均': f"{rf_mean:.4f}",
                'IRL平均': f"{irl_mean:.4f}",
                '差分': f"{diff:+.4f}",
                '差分%': f"{diff_pct:+.2f}%",
                'IRL勝ち': wins,
                '引き分け': ties,
                'RF勝ち': losses
            })

        summary_df = pd.DataFrame(summary_data)
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n")

        f.write("## パターン別詳細比較\n\n")

        # パターン別の詳細テーブル
        detail_data = []
        for _, row in df_merged.iterrows():
            detail_data.append({
                'Pattern': row['pattern'],
                'F1_RF': f"{row['f1_RF']:.4f}",
                'F1_IRL': f"{row['f1_IRL']:.4f}",
                'AUC-ROC_RF': f"{row['auc_roc_RF']:.4f}",
                'AUC-ROC_IRL': f"{row['auc_roc_IRL']:.4f}",
                'Precision_RF': f"{row['precision_RF']:.4f}",
                'Precision_IRL': f"{row['precision_IRL']:.4f}",
                'Recall_RF': f"{row['recall_RF']:.4f}",
                'Recall_IRL': f"{row['recall_IRL']:.4f}",
            })

        detail_df = pd.DataFrame(detail_data)
        f.write(detail_df.to_markdown(index=False))
        f.write("\n\n")

        f.write("## 分析結果\n\n")

        # 総合評価
        overall_winner = "IRL" if df_merged['f1_IRL'].mean() > df_merged['f1_RF'].mean() else "RF"
        f.write(f"### 総合評価（F1スコア平均）\n\n")
        f.write(f"- **勝者**: {overall_winner}\n")
        f.write(f"- RF平均F1: {df_merged['f1_RF'].mean():.4f}\n")
        f.write(f"- IRL平均F1: {df_merged['f1_IRL'].mean():.4f}\n\n")

        # 各メトリクスでの優位性
        f.write("### メトリクス別優位性\n\n")
        for metric in metrics:
            rf_col = f"{metric}_RF"
            irl_col = f"{metric}_IRL"
            winner = "IRL" if df_merged[irl_col].mean() > df_merged[rf_col].mean() else "RF"
            f.write(f"- **{metric.upper()}**: {winner} が優位\n")
        f.write("\n")

        f.write("## 可視化\n\n")
        f.write("- `comparison_barplots.png`: 各パターンでのメトリクス比較（バープロット）\n")
        f.write("- `comparison_scatterplots.png`: RF vs IRL 散布図\n")
        f.write("- `comparison_diff_heatmap.png`: 差分ヒートマップ（IRL - RF）\n\n")

    logger.info(f"✓ 比較レポート保存: {report_path}")


def main():
    logger.info("=" * 80)
    logger.info("RF vs IRL 10パターン比較分析")
    logger.info("=" * 80)

    # 出力ディレクトリ
    output_dir = Path("/Users/kazuki-h/research/multiproject_research/outputs/comparison_rf_irl_10patterns")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. IRL結果読み込み（16パターン）
    irl_results = load_irl_results()

    # 2. IRL結果から10パターンを抽出
    irl_10_results = extract_10_patterns(irl_results)

    # 3. RF結果読み込み（10パターン）
    rf_results = load_rf_results()

    # 4. 比較分析
    df_merged = compare_results(rf_results, irl_10_results)

    # 5. 比較データをCSVで保存
    csv_path = output_dir / "comparison_data.csv"
    df_merged.to_csv(csv_path, index=False)
    logger.info(f"✓ 比較データ保存: {csv_path}")

    # 6. 可視化
    create_comparison_visualizations(df_merged, output_dir)

    # 7. レポート作成
    save_comparison_report(df_merged, output_dir)

    logger.info("\n✓ 完了！")
    logger.info(f"出力ディレクトリ: {output_dir}")


if __name__ == '__main__':
    main()

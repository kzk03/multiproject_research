#!/usr/bin/env python3
"""
クロス時間評価結果のヒートマップ作成スクリプト
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
plt.rcParams['font.family'] = 'sans-serif'


def create_single_heatmap(matrix: pd.DataFrame, title: str, output_path: Path,
                          vmin: float = None, vmax: float = None, cmap: str = 'YlOrRd'):
    """
    単一メトリクスのヒートマップを作成

    Args:
        matrix: メトリクスマトリクス
        title: タイトル
        output_path: 出力パス
        vmin: カラーバーの最小値
        vmax: カラーバーの最大値
        cmap: カラーマップ
    """
    plt.figure(figsize=(10, 8))

    # ヒートマップを描画
    sns.heatmap(
        matrix.astype(float),
        annot=True,
        fmt='.3f',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': 'Score'},
        linewidths=0.5,
        linecolor='gray'
    )

    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('評価期間', fontsize=14)
    plt.ylabel('訓練期間', fontsize=14)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"保存: {output_path}")


def create_combined_heatmap(matrices: dict, output_path: Path):
    """
    4つのメトリクスを統合したヒートマップを作成

    Args:
        matrices: メトリクスマトリクスの辞書
        output_path: 出力パス
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('クロス時間評価: 4メトリクス統合ヒートマップ', fontsize=20, y=0.995)

    metrics_info = [
        ('AUC_ROC', 'AUC-ROC', 'RdYlGn', 0.5, 1.0),
        ('AUC_PR', 'AUC-PR', 'RdYlGn', 0.5, 1.0),
        ('PRECISION', 'Precision', 'Blues', 0.0, 1.0),
        ('RECALL', 'Recall', 'Oranges', 0.0, 1.0)
    ]

    for idx, (metric_key, metric_name, cmap, vmin, vmax) in enumerate(metrics_info):
        ax = axes[idx // 2, idx % 2]

        if metric_key in matrices:
            matrix = matrices[metric_key].astype(float)

            sns.heatmap(
                matrix,
                annot=True,
                fmt='.3f',
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                cbar_kws={'label': 'Score'},
                linewidths=0.5,
                linecolor='gray',
                ax=ax
            )

            ax.set_title(metric_name, fontsize=16, pad=15)
            ax.set_xlabel('評価期間', fontsize=14)
            ax.set_ylabel('訓練期間', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"保存: {output_path}")


def create_summary_statistics(matrices: dict, output_path: Path):
    """
    サマリー統計を作成

    Args:
        matrices: メトリクスマトリクスの辞書
        output_path: 出力パス
    """
    summary = {}

    for metric_name, matrix in matrices.items():
        # 数値に変換
        matrix_values = matrix.astype(float).values

        # NaNを除外
        valid_values = matrix_values[~np.isnan(matrix_values)]

        if len(valid_values) > 0:
            summary[metric_name] = {
                'mean': float(np.mean(valid_values)),
                'std': float(np.std(valid_values)),
                'min': float(np.min(valid_values)),
                'max': float(np.max(valid_values)),
                'median': float(np.median(valid_values))
            }

            # 対角線の統計（同期間評価）
            diagonal_values = []
            for i in range(min(matrix.shape)):
                val = matrix.iloc[i, i]
                if not np.isnan(val):
                    diagonal_values.append(val)

            if diagonal_values:
                summary[metric_name]['diagonal_mean'] = float(np.mean(diagonal_values))
                summary[metric_name]['diagonal_std'] = float(np.std(diagonal_values))

    # 保存
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"保存: {output_path}")

    # サマリーを表示
    print("\n=== サマリー統計 ===")
    for metric_name, stats in summary.items():
        print(f"\n{metric_name}:")
        print(f"  平均: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  範囲: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"  中央値: {stats['median']:.4f}")
        if 'diagonal_mean' in stats:
            print(f"  対角線平均: {stats['diagonal_mean']:.4f} ± {stats['diagonal_std']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="クロス時間評価結果のヒートマップ作成"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="クロス評価結果ディレクトリ"
    )

    args = parser.parse_args()

    input_dir = Path(args.input)

    # 出力ディレクトリ
    heatmaps_dir = input_dir / "heatmaps"
    heatmaps_dir.mkdir(exist_ok=True)

    # メトリクスマトリクスを読み込み
    print("=" * 80)
    print("メトリクスマトリクスを読み込み中...")
    print("=" * 80)

    matrices = {}
    metric_files = list(input_dir.glob("matrix_*.csv"))

    for matrix_file in metric_files:
        metric_name = matrix_file.stem.replace('matrix_', '')
        matrix = pd.read_csv(matrix_file, index_col=0)
        matrices[metric_name] = matrix
        print(f"読み込み: {matrix_file.name}")

    if not matrices:
        print("エラー: メトリクスマトリクスが見つかりません")
        return

    # 個別ヒートマップを作成
    print("\n" + "=" * 80)
    print("個別ヒートマップを作成中...")
    print("=" * 80)

    heatmap_configs = {
        'AUC_ROC': ('AUC-ROC ヒートマップ', 'RdYlGn', 0.5, 1.0),
        'AUC_PR': ('AUC-PR ヒートマップ', 'RdYlGn', 0.5, 1.0),
        'PRECISION': ('Precision ヒートマップ', 'Blues', 0.0, 1.0),
        'RECALL': ('Recall ヒートマップ', 'Oranges', 0.0, 1.0),
        'f1_score': ('F1 Score ヒートマップ', 'Purples', 0.0, 1.0)
    }

    for metric_name, (title, cmap, vmin, vmax) in heatmap_configs.items():
        if metric_name in matrices:
            output_path = heatmaps_dir / f"heatmap_{metric_name}.png"
            create_single_heatmap(
                matrices[metric_name],
                title,
                output_path,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap
            )

    # 統合ヒートマップを作成
    print("\n" + "=" * 80)
    print("統合ヒートマップを作成中...")
    print("=" * 80)

    combined_path = heatmaps_dir / "heatmap_4_metrics.png"
    create_combined_heatmap(matrices, combined_path)

    # サマリー統計を作成
    print("\n" + "=" * 80)
    print("サマリー統計を作成中...")
    print("=" * 80)

    summary_path = input_dir / "summary_statistics.json"
    create_summary_statistics(matrices, summary_path)

    print("\n" + "=" * 80)
    print("ヒートマップ作成完了")
    print(f"出力ディレクトリ: {heatmaps_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

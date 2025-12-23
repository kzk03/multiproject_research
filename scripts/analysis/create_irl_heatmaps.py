#!/usr/bin/env python3
"""
IRL Nova予測結果のヒートマップ作成スクリプト

RFと同じ形式でIRLの10パターン予測結果をヒートマップ化する。
- 横軸：訓練期間（左から0-3m → 9-12m）
- 縦軸：評価期間（下から0-3m → 9-12m）
- 10パターン（train ≤ eval）のみ表示

使用方法:
    uv run python scripts/analysis/create_irl_heatmaps.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
plt.rcParams['font.family'] = 'sans-serif'


# パス設定
BASE_DIR = Path(__file__).parent.parent.parent
IRL_RESULTS_DIR = BASE_DIR / "results" / "review_continuation_cross_eval_nova"
OUTPUT_DIR = IRL_RESULTS_DIR / "heatmaps"

# 期間ラベル
PERIODS = ['0-3m', '3-6m', '6-9m', '9-12m']


def build_metrics_matrices() -> dict:
    """
    IRL結果から各メトリクスのマトリクスを構築
    
    Returns:
        メトリクス名をキーとした4x4マトリクスの辞書
        行: 評価期間（下から上: 0-3m → 9-12m）
        列: 訓練期間（左から右: 0-3m → 9-12m）
    """
    metrics = ['precision', 'recall', 'f1', 'auc_roc']
    # 評価期間を行（逆順で下から上）、訓練期間を列
    eval_periods_reversed = PERIODS[::-1]  # 下から上に表示するため逆順
    matrices = {m: pd.DataFrame(
        np.nan, 
        index=eval_periods_reversed,  # 行: 評価期間（逆順）
        columns=PERIODS               # 列: 訓練期間
    ) for m in metrics}
    
    # 10パターン（train <= eval）のみ読み込み
    for train_period in PERIODS:
        for eval_period in PERIODS:
            train_idx = PERIODS.index(train_period)
            eval_idx = PERIODS.index(eval_period)
            
            # train <= eval の場合のみ
            if train_idx <= eval_idx:
                metrics_file = IRL_RESULTS_DIR / f"train_{train_period}" / f"eval_{eval_period}" / "metrics.json"
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        data = json.load(f)
                        matrices['precision'].loc[eval_period, train_period] = data.get('precision', np.nan)
                        matrices['recall'].loc[eval_period, train_period] = data.get('recall', np.nan)
                        matrices['f1'].loc[eval_period, train_period] = data.get('f1_score', np.nan)
                        matrices['auc_roc'].loc[eval_period, train_period] = data.get('auc_roc', np.nan)
    return matrices


def create_single_heatmap(matrix: pd.DataFrame, title: str, output_path: Path,
                          vmin: float = None, vmax: float = None):
    """
    単一メトリクスのヒートマップを作成
    """
    plt.figure(figsize=(10, 8))

    # 有効なセルのみ表示
    mask = matrix.isna()

    sns.heatmap(
        matrix.astype(float),
        annot=True,
        fmt='.3f',
        cmap='Blues',
        vmin=vmin,
        vmax=vmax,
        mask=mask,
        annot_kws={'size': 18, 'weight': 'bold'},
        cbar_kws={'label': 'Score'},
        linewidths=0.5,
        linecolor='gray'
    )

    plt.title(title, fontsize=20, pad=20)
    plt.xlabel('訓練期間', fontsize=16)
    plt.ylabel('評価期間', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"保存: {output_path}")


def create_4_metrics_heatmap(matrices: dict, output_path: Path):
    """
    4つのメトリクス(F1, AUC-ROC, Precision, Recall)を統合したヒートマップを作成
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('クロス時間評価ヒートマップ', 
                 fontsize=22, y=0.995)

    metrics_info = [
        ('precision', 'Precision', 0.4, 1.0),
        ('recall', 'Recall', 0.4, 1.0),
        ('f1', 'F1 Score', 0.4, 0.9),
        ('auc_roc', 'AUC-ROC', 0.5, 1.0),
    ]

    for idx, (metric_key, metric_name, vmin, vmax) in enumerate(metrics_info):
        ax = axes[idx // 2, idx % 2]

        if metric_key in matrices:
            matrix = matrices[metric_key]
            mask = matrix.isna()

            sns.heatmap(
                matrix.astype(float),
                annot=True,
                fmt='.3f',
                cmap='Blues',
                vmin=vmin,
                vmax=vmax,
                mask=mask,
                annot_kws={'size': 16, 'weight': 'bold'},
                cbar_kws={'label': 'Score'},
                linewidths=0.5,
                linecolor='gray',
                ax=ax
            )

            ax.set_title(metric_name, fontsize=18, pad=15)
            ax.set_xlabel('訓練期間', fontsize=14)
            ax.set_ylabel('評価期間', fontsize=14)
            ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"保存: {output_path}")


def main():
    """メイン処理"""
    # 出力ディレクトリ作成
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # マトリクス構築
    matrices = build_metrics_matrices()
    print(f"IRL結果読み込み完了: 10パターン")
    
    # 個別ヒートマップ作成
    metrics_config = [
        ('precision', 'IRL Nova: Precision', 0.4, 1.0),
        ('recall', 'IRL Nova: Recall', 0.4, 1.0),
        ('f1', 'IRL Nova: F1 Score', 0.4, 0.9),
        ('auc_roc', 'IRL Nova: AUC-ROC', 0.5, 1.0)
    ]
    
    for metric_key, title, vmin, vmax in metrics_config:
        output_file = OUTPUT_DIR / f"heatmap_{metric_key}.png"
        create_single_heatmap(matrices[metric_key], title, output_file, vmin, vmax)
    
    # 4メトリクス統合ヒートマップ
    create_4_metrics_heatmap(matrices, OUTPUT_DIR / "heatmap_4_metrics.png")
    
    print(f"\n完了！出力先: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

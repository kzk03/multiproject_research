#!/usr/bin/env python3
"""
RF Case2 予測結果のヒートマップ作成スクリプト

RFの10パターン予測結果をIRLと同様の形式でヒートマップ化する。
訓練期間(行)×評価期間(列)のマトリクス形式で各メトリクスを表示。

使用方法:
    uv run python scripts/analysis/create_rf_heatmaps.py
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
RF_RESULTS = BASE_DIR / "outputs" / "singleproject" /"rf_nova_case2_simple" / "results.json"
OUTPUT_DIR = BASE_DIR / "outputs" / "singleproject" /"rf_nova_case2_simple" / "heatmaps"

# 期間ラベル
PERIODS = ['0-3m', '3-6m', '6-9m', '9-12m']


def load_rf_results(results_path: Path) -> list:
    """RF結果JSONを読み込む"""
    with open(results_path, 'r') as f:
        return json.load(f)


def build_metrics_matrices(results: list) -> dict:
    """
    RF結果から各メトリクスのマトリクスを構築
    
    Args:
        results: RF結果のリスト
        
    Returns:
        メトリクス名をキーとした4x4マトリクスの辞書
        行: 評価期間（下から上: 0-3m → 9-12m）
        列: 訓練期間（左から右: 0-3m → 9-12m）
    """
    metrics = ['f1', 'auc_roc', 'precision', 'recall']
    # 評価期間を行（逆順で下から上）、訓練期間を列
    eval_periods_reversed = PERIODS[::-1]  # 下から上に表示するため逆順
    matrices = {m: pd.DataFrame(
        np.nan, 
        index=eval_periods_reversed,  # 行: 評価期間（逆順）
        columns=PERIODS               # 列: 訓練期間
    ) for m in metrics}
    
    for r in results:
        pattern = r['pattern']
        # "0-3m → 0-3m" から訓練期間と評価期間を抽出
        parts = pattern.replace('→', '→').replace('→', '→').split('→')
        if len(parts) != 2:
            parts = pattern.split(' → ')
        train_period = parts[0].strip()
        eval_period = parts[1].strip()
        
        # マトリクスに値を設定（行=評価期間、列=訓練期間）
        for metric in metrics:
            if train_period in PERIODS and eval_period in PERIODS:
                matrices[metric].loc[eval_period, train_period] = r[metric]
    
    return matrices


def create_single_heatmap(matrix: pd.DataFrame, title: str, output_path: Path,
                          vmin: float = None, vmax: float = None, cmap: str = 'Blues'):
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
    fig.suptitle('RF:クロス時間評価ヒートマップ', 
                 fontsize=22, y=0.995)

    metrics_info = [
        ('f1', 'F1 Score', 'Blues', 0.4, 0.9),
        ('auc_roc', 'AUC-ROC', 'Blues', 0.5, 1.0),
        ('precision', 'Precision', 'Blues', 0.4, 1.0),
        ('recall', 'Recall', 'Blues', 0.4, 1.0)
    ]

    for idx, (metric_key, metric_name, cmap, vmin, vmax) in enumerate(metrics_info):
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


def create_comparison_heatmap_with_irl(rf_matrices: dict, irl_dir: Path, output_path: Path):
    """
    RF と IRL の比較ヒートマップを作成
    軸: 横=訓練期間(左から0-3m)、縦=評価期間(下から0-3m)
    """
    # IRLの結果を読み込む
    irl_results_dir = BASE_DIR / "results" / "review_continuation_cross_eval_nova"
    
    # IRL結果を収集（行=評価期間(逆順)、列=訓練期間）
    eval_periods_reversed = PERIODS[::-1]
    irl_matrices = {m: pd.DataFrame(np.nan, index=eval_periods_reversed, columns=PERIODS) 
                    for m in ['f1', 'auc_roc', 'precision', 'recall']}
    
    for train_period in PERIODS:
        for eval_period in PERIODS:
            train_idx = PERIODS.index(train_period)
            eval_idx = PERIODS.index(eval_period)
            
            # train <= eval の場合のみ
            if train_idx <= eval_idx:
                metrics_file = irl_results_dir / f"train_{train_period}" / f"eval_{eval_period}" / "metrics.json"
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                        irl_matrices['f1'].loc[eval_period, train_period] = metrics.get('f1_score', np.nan)
                        irl_matrices['auc_roc'].loc[eval_period, train_period] = metrics.get('auc_roc', np.nan)
                        irl_matrices['precision'].loc[eval_period, train_period] = metrics.get('precision', np.nan)
                        irl_matrices['recall'].loc[eval_period, train_period] = metrics.get('recall', np.nan)
    
    # 比較ヒートマップを作成（F1とAUC-ROCの2つ）
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('IRL vs RF Case2: Nova クロス時間評価比較', fontsize=24, y=0.995)

    comparisons = [
        ('f1', 'F1 Score', 'Blues', 0.4, 0.9),
        ('auc_roc', 'AUC-ROC', 'Blues', 0.5, 1.0),
    ]

    for col_idx, (metric_key, metric_name, cmap, vmin, vmax) in enumerate(comparisons):
        # IRL
        ax = axes[0, col_idx]
        irl_matrix = irl_matrices[metric_key]
        mask = irl_matrix.isna()
        
        sns.heatmap(
            irl_matrix.astype(float),
            annot=True,
            fmt='.3f',
            cmap='Blues',
            vmin=vmin,
            vmax=vmax,
            mask=mask,
            annot_kws={'size': 16, 'weight': 'bold'},
            linewidths=0.5,
            linecolor='gray',
            ax=ax
        )
        ax.set_title(f'IRL: {metric_name}', fontsize=20, pad=15)
        ax.set_xlabel('訓練期間', fontsize=16)
        ax.set_ylabel('評価期間', fontsize=16)
        ax.tick_params(labelsize=12)
        
        # RF
        ax = axes[1, col_idx]
        rf_matrix = rf_matrices[metric_key]
        mask = rf_matrix.isna()
        
        sns.heatmap(
            rf_matrix.astype(float),
            annot=True,
            fmt='.3f',
            cmap='Blues',
            vmin=vmin,
            vmax=vmax,
            mask=mask,
            annot_kws={'size': 16, 'weight': 'bold'},
            linewidths=0.5,
            linecolor='gray',
            ax=ax
        )
        ax.set_title(f'RF Case2: {metric_name}', fontsize=20, pad=15)
        ax.set_xlabel('訓練期間', fontsize=16)
        ax.set_ylabel('評価期間', fontsize=16)
        ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"保存: {output_path}")


def main():
    """メイン処理"""
    # 出力ディレクトリ作成
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # RF結果読み込み
    results = load_rf_results(RF_RESULTS)
    print(f"RF結果読み込み完了: {len(results)}パターン")
    
    # マトリクス構築
    matrices = build_metrics_matrices(results)
    
    # 個別ヒートマップ作成
    metrics_config = [
        ('f1', 'RF Case2: F1 Score', 'Blues', 0.4, 0.9),
        ('auc_roc', 'RF Case2: AUC-ROC', 'Blues', 0.5, 1.0),
        ('precision', 'RF Case2: Precision', 'Blues', 0.4, 1.0),
        ('recall', 'RF Case2: Recall', 'Blues', 0.4, 1.0)
    ]
    
    for metric_key, title, cmap, vmin, vmax in metrics_config:
        output_file = OUTPUT_DIR / f"heatmap_{metric_key}.png"
        create_single_heatmap(matrices[metric_key], title, output_file, vmin, vmax, cmap)
    
    # 4メトリクス統合ヒートマップ
    create_4_metrics_heatmap(matrices, OUTPUT_DIR / "heatmap_4_metrics.png")
    
    # IRL vs RF 比較ヒートマップ
    create_comparison_heatmap_with_irl(
        matrices, 
        BASE_DIR / "results" / "review_continuation_cross_eval_nova" / "heatmaps",
        OUTPUT_DIR / "heatmap_irl_rf_comparison.png"
    )
    
    print(f"\n完了！出力先: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

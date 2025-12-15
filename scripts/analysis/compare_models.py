#!/usr/bin/env python3
"""
Phase 2: 基本モデル比較分析

Nova単体 vs 20proj vs 50proj vs 50proj_improved の予測性能を比較し、
予測の一致・不一致パターンを分析する。
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib_venn import venn3

# パス設定
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style("whitegrid")


def load_model_metrics(model_name: str, base_dir: Path) -> Dict:
    """モデルのメトリクスを読み込み"""
    
    metrics_list = []
    
    # 対角線要素（同期間訓練・評価）を収集
    for period in ['0-3m', '3-6m', '6-9m', '9-12m']:
        metrics_file = base_dir / f'train_{period}' / f'eval_{period}' / 'metrics.json'
        if metrics_file.exists():
            with open(metrics_file) as f:
                m = json.load(f)
                m['period'] = period
                metrics_list.append(m)
    
    if not metrics_list:
        return None
    
    # 平均を計算
    avg_metrics = {
        'model': model_name,
        'f1': np.mean([m['f1_score'] for m in metrics_list]),
        'precision': np.mean([m['precision'] for m in metrics_list]),
        'recall': np.mean([m['recall'] for m in metrics_list]),
        'auc_roc': np.mean([m['auc_roc'] for m in metrics_list]),
        'auc_pr': np.mean([m['auc_pr'] for m in metrics_list]),
        'periods': metrics_list
    }
    
    return avg_metrics


def collect_all_model_results() -> pd.DataFrame:
    """全モデルの結果を収集"""
    
    results = []
    
    # Nova単体（結果がある場合）
    nova_dir = ROOT / 'results' / 'review_continuation_cross_eval_nova'
    if nova_dir.exists():
        # Novaの結果を探す
        pass  # TODO: Nova単体の結果パスを確認
    
    # 20プロジェクト
    proj20_dir = ROOT / 'outputs' / 'multiproject_irl_full'
    if proj20_dir.exists():
        metrics_file = proj20_dir / 'metrics.json'
        if metrics_file.exists():
            with open(metrics_file) as f:
                m = json.load(f)
            results.append({
                'model': '20 Projects',
                'f1': m['f1_score'],
                'precision': m['precision'],
                'recall': m['recall'],
                'auc_roc': m['auc_roc'],
                'auc_pr': m['auc_pr']
            })
    
    # 50プロジェクト（元）
    for exp in ['no_os', '2x_os', '3x_os']:
        base_dir = ROOT / 'outputs' / '50projects_irl' / exp
        if base_dir.exists():
            m = load_model_metrics(f'50proj ({exp})', base_dir)
            if m:
                results.append(m)
    
    # 50プロジェクト（改善版）
    for exp in ['no_os', '2x_os', '3x_os']:
        base_dir = ROOT / 'outputs' / '50projects_irl_improved' / exp
        if base_dir.exists():
            m = load_model_metrics(f'50proj_improved ({exp})', base_dir)
            if m:
                results.append(m)
    
    df = pd.DataFrame(results)
    return df


def plot_model_comparison_radar(df: pd.DataFrame, output_path: Path):
    """モデル比較レーダーチャート"""
    
    metrics = ['f1', 'precision', 'recall', 'auc_roc', 'auc_pr']
    
    # 上位5モデルを選択
    top_models = df.nlargest(5, 'f1')
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    for _, row in top_models.iterrows():
        values = [row[m] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=row['model'])
        ax.fill(angles, values, alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['F1', 'Precision', 'Recall', 'AUC-ROC', 'AUC-PR'])
    ax.set_ylim(0, 1)
    ax.set_title('Model Performance Comparison (Radar Chart)', size=16, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved radar chart: {output_path}")


def plot_metric_comparison_bars(df: pd.DataFrame, output_path: Path):
    """メトリクス別の棒グラフ比較"""
    
    metrics = ['f1', 'precision', 'recall', 'auc_roc', 'auc_pr']
    metric_names = ['F1 Score', 'Precision', 'Recall', 'AUC-ROC', 'AUC-PR']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]
        
        # モデル名を短縮
        df_plot = df.copy()
        df_plot['model_short'] = df_plot['model'].str.replace('50proj_improved', '50p_v2').str.replace('50proj', '50p')
        
        # 降順ソート
        df_sorted = df_plot.sort_values(metric, ascending=False)
        
        bars = ax.barh(range(len(df_sorted)), df_sorted[metric])
        
        # カラーマップ
        colors = plt.cm.viridis(df_sorted[metric] / df_sorted[metric].max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels(df_sorted['model_short'], fontsize=9)
        ax.set_xlabel(name, fontsize=11)
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3)
        
        # 値を表示
        for i, (idx_row, row) in enumerate(df_sorted.iterrows()):
            ax.text(row[metric] + 0.01, i, f'{row[metric]:.3f}', 
                   va='center', fontsize=9)
    
    # 最後のサブプロット（6番目）を使ってサマリーテーブル
    axes[5].axis('off')
    
    # 最高値を表示
    best_models = []
    for metric, name in zip(metrics, metric_names):
        best_row = df.loc[df[metric].idxmax()]
        best_models.append([name, best_row['model'], f"{best_row[metric]:.3f}"])
    
    table = axes[5].table(cellText=best_models, 
                         colLabels=['Metric', 'Best Model', 'Score'],
                         cellLoc='left',
                         loc='center',
                         colWidths=[0.3, 0.5, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    axes[5].set_title('Best Models by Metric', fontsize=12, pad=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved metric comparison: {output_path}")


def create_comparison_summary_table(df: pd.DataFrame, output_path: Path):
    """比較サマリーテーブルをCSVとして保存"""
    
    # ソート
    df_sorted = df.sort_values('f1', ascending=False)
    
    # 出力用に整形
    df_out = df_sorted[['model', 'f1', 'precision', 'recall', 'auc_roc', 'auc_pr']].copy()
    df_out.columns = ['Model', 'F1', 'Precision', 'Recall', 'AUC-ROC', 'AUC-PR']
    
    # 保存
    df_out.to_csv(output_path, index=False, float_format='%.4f')
    print(f"Saved summary table: {output_path}")
    
    # コンソールにも表示
    print("\n" + "="*80)
    print("Model Comparison Summary")
    print("="*80)
    print(df_out.to_string(index=False))
    print("="*80 + "\n")
    
    return df_out


def main():
    print("="*80)
    print("Phase 2: Model Comparison Analysis")
    print("="*80)
    print()
    
    # 結果収集
    print("[1/4] Collecting model results...")
    df = collect_all_model_results()
    
    if df.empty:
        print("No model results found!")
        return
    
    print(f"Found {len(df)} model configurations")
    print()
    
    # 出力ディレクトリ
    output_dir = ROOT / 'outputs' / 'analysis_data'
    vis_dir = ROOT / 'outputs' / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # サマリーテーブル
    print("[2/4] Creating summary table...")
    create_comparison_summary_table(
        df, 
        output_dir / 'model_comparison_summary.csv'
    )
    
    # レーダーチャート
    print("[3/4] Creating radar chart...")
    plot_model_comparison_radar(
        df,
        vis_dir / 'model_comparison_radar.png'
    )
    
    # メトリクス別棒グラフ
    print("[4/4] Creating metric comparison charts...")
    plot_metric_comparison_bars(
        df,
        vis_dir / 'model_comparison_bars.png'
    )
    
    print()
    print("="*80)
    print("Phase 2 Analysis Complete!")
    print("="*80)
    print(f"Results saved to: {output_dir}")
    print(f"Visualizations saved to: {vis_dir}")
    print()


if __name__ == '__main__':
    main()

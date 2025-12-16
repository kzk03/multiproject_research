#!/usr/bin/env python3
"""
Phase 3: 予測的中開発者の特性分析

開発者ごとの予測的中率を計算し、14次元状態特徴量との関係を分析する。
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# パス設定
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")


# 14次元状態特徴量の定義
STATE_FEATURES = [
    'experience_days',
    'total_changes',
    'total_reviews',
    'recent_activity_frequency',
    'avg_activity_gap',
    'collaboration_score',
    'code_quality_score',
    'recent_acceptance_rate',
    'review_load',
    'activity_trend',
    # マルチプロジェクト特徴量
    'project_count',
    'project_activity_distribution',
    'main_project_contribution_ratio',
    'cross_project_collaboration_score'
]


def load_predictions_and_labels(model_dir: Path) -> pd.DataFrame:
    """予測結果と正解ラベルを読み込み"""
    
    all_predictions = []
    
    # 全時間窓の予測を収集
    for train_period in ['0-3m', '3-6m', '6-9m', '9-12m']:
        for eval_period in ['0-3m', '3-6m', '6-9m', '9-12m']:
            pred_file = model_dir / f'train_{train_period}' / f'eval_{eval_period}' / 'predictions.csv'
            
            if pred_file.exists():
                df = pd.read_csv(pred_file)
                df['train_period'] = train_period
                df['eval_period'] = eval_period
                all_predictions.append(df)
    
    if not all_predictions:
        return pd.DataFrame()
    
    return pd.concat(all_predictions, ignore_index=True)


def calculate_developer_accuracy(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """開発者ごとの予測的中率を計算"""
    
    if 'developer_id' not in predictions_df.columns:
        # reviewer_emailをdeveloper_idとして使用
        predictions_df['developer_id'] = predictions_df.get('reviewer_email', 
                                                              predictions_df.index)
    
    # 予測結果を二値化（閾値0.5）
    if 'predicted_binary' in predictions_df.columns:
        predictions_df['pred_binary'] = predictions_df['predicted_binary']
    elif 'predicted_prob' in predictions_df.columns:
        predictions_df['pred_binary'] = (predictions_df['predicted_prob'] > 0.5).astype(int)
    elif 'prediction' in predictions_df.columns:
        predictions_df['pred_binary'] = (predictions_df['prediction'] > 0.5).astype(int)
    elif 'predicted_label' in predictions_df.columns:
        predictions_df['pred_binary'] = predictions_df['predicted_label']
    else:
        print("Warning: No prediction column found")
        print(f"Available columns: {predictions_df.columns.tolist()}")
        return pd.DataFrame()

    # 正解ラベル
    if 'true_label' in predictions_df.columns:
        label_col = 'true_label'
    elif 'label' in predictions_df.columns:
        label_col = 'label'
    else:
        print("Warning: No label column found")
        print(f"Available columns: {predictions_df.columns.tolist()}")
        return pd.DataFrame()
    
    # 開発者ごとに集計
    developer_stats = predictions_df.groupby('developer_id').agg({
        'pred_binary': 'count',  # 総予測数
        label_col: lambda x: ((predictions_df.loc[x.index, 'pred_binary'] == x).sum())  # 正解数
    }).rename(columns={
        'pred_binary': 'total_predictions',
        label_col: 'correct_predictions'
    })
    
    developer_stats['accuracy'] = (
        developer_stats['correct_predictions'] / developer_stats['total_predictions']
    )
    
    # セグメント分類
    developer_stats['segment'] = pd.cut(
        developer_stats['accuracy'],
        bins=[0, 0.5, 0.8, 0.95, 1.0, 1.01],
        labels=['Unpredictable', 'Low', 'Medium', 'High', 'Perfect'],
        include_lowest=True
    )
    
    return developer_stats


def load_developer_features(data_path: Path) -> pd.DataFrame:
    """開発者の特徴量データを読み込み"""
    
    # 50プロジェクトのデータから特徴量を抽出
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(data_path)
    
    # reviewer_emailをdeveloper_idとして使用
    df['developer_id'] = df['reviewer_email']
    
    # 利用可能な特徴量のみを抽出
    available_features = [f for f in STATE_FEATURES if f in df.columns]
    
    if not available_features:
        print("Warning: No state features found in data")
        return pd.DataFrame()
    
    # 開発者ごとに集計（最新の値を使用）
    df_features = df.sort_values('request_time').groupby('developer_id').last()[available_features]
    
    return df_features


def analyze_segment_characteristics(
    developer_stats: pd.DataFrame,
    developer_features: pd.DataFrame
) -> pd.DataFrame:
    """セグメント別の特徴量分析"""
    
    # 統合
    merged = developer_stats.join(developer_features, how='inner')
    
    if merged.empty:
        print("No matching developers found")
        return pd.DataFrame()
    
    # セグメント別の平均
    segment_means = merged.groupby('segment')[developer_features.columns].mean()
    
    # 全体平均
    overall_mean = merged[developer_features.columns].mean()
    
    # 差分（標準化）
    segment_diff = segment_means.sub(overall_mean, axis=1).div(
        merged[developer_features.columns].std(), axis=1
    )
    
    return segment_means, segment_diff, merged


def plot_segment_heatmap(segment_diff: pd.DataFrame, output_path: Path):
    """セグメント別特徴量のヒートマップ"""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sns.heatmap(
        segment_diff.T,
        cmap='RdBu_r',
        center=0,
        annot=True,
        fmt='.2f',
        cbar_kws={'label': 'Standardized Difference from Mean'},
        ax=ax
    )
    
    ax.set_title('Developer Segment Characteristics (Standardized)', fontsize=16, pad=20)
    ax.set_xlabel('Prediction Accuracy Segment', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved segment heatmap: {output_path}")


def plot_pca_scatter(merged_df: pd.DataFrame, output_path: Path):
    """PCA 2次元散布図（的中率でカラーマップ）"""
    
    feature_cols = [c for c in STATE_FEATURES if c in merged_df.columns]
    
    # 欠損値処理
    X = merged_df[feature_cols].fillna(0)
    
    # 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # プロット
    fig, ax = plt.subplots(figsize=(12, 10))
    
    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=merged_df['accuracy'],
        cmap='viridis',
        s=100,
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Prediction Accuracy', fontsize=12)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax.set_title('Developer Distribution in Feature Space (PCA)', fontsize=16, pad=20)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved PCA scatter plot: {output_path}")


def analyze_project_count_effect(merged_df: pd.DataFrame, output_path: Path):
    """プロジェクト数と予測精度の関係を分析"""
    
    if 'project_count' not in merged_df.columns:
        print("project_count not available")
        return
    
    # プロジェクトタイプ分類
    def classify_project_type(count):
        if count == 1:
            return 'Specialist (1 proj)'
        elif count <= 3:
            return 'Contributor (2-3 proj)'
        else:
            return 'Expert (4+ proj)'
    
    merged_df['project_type'] = merged_df['project_count'].apply(classify_project_type)
    
    # タイプ別統計
    project_type_stats = merged_df.groupby('project_type').agg({
        'accuracy': ['mean', 'std', 'count'],
        'project_count': 'mean'
    }).round(3)
    
    print("\n" + "="*80)
    print("Project Type vs Prediction Accuracy")
    print("="*80)
    print(project_type_stats)
    print("="*80 + "\n")
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # バイオリンプロット
    ax = axes[0]
    sns.violinplot(
        data=merged_df,
        x='project_type',
        y='accuracy',
        ax=ax,
        palette='Set2'
    )
    ax.set_title('Accuracy Distribution by Project Type', fontsize=14)
    ax.set_xlabel('Project Type', fontsize=12)
    ax.set_ylabel('Prediction Accuracy', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # カウント別散布図
    ax = axes[1]
    scatter = ax.scatter(
        merged_df['project_count'],
        merged_df['accuracy'],
        alpha=0.5,
        s=100,
        edgecolors='black',
        linewidth=0.5
    )
    
    # トレンドライン
    z = np.polyfit(merged_df['project_count'], merged_df['accuracy'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(merged_df['project_count'].min(), 
                          merged_df['project_count'].max(), 100)
    ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Trend')
    
    ax.set_title('Accuracy vs Project Count', fontsize=14)
    ax.set_xlabel('Number of Projects', fontsize=12)
    ax.set_ylabel('Prediction Accuracy', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved project type analysis: {output_path}")
    
    # 統計をCSVに保存
    csv_path = output_path.parent.parent / 'analysis_data' / 'project_type_stats.csv'
    project_type_stats.to_csv(csv_path)
    print(f"Saved project type stats: {csv_path}")


def main():
    print("="*80)
    print("Phase 3: Developer Characteristics Analysis")
    print("="*80)
    print()
    
    # モデル選択（50proj no_osを使用）
    model_dir = ROOT / 'outputs' / '50projects_irl' / 'no_os'
    data_path = ROOT / 'data' / 'openstack_50proj_2021_2024_feat.csv'
    
    # 出力ディレクトリ
    output_dir = ROOT / 'outputs' / 'analysis_data'
    vis_dir = ROOT / 'outputs' / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # [1] 予測結果読み込み
    print("[1/6] Loading predictions...")
    predictions_df = load_predictions_and_labels(model_dir)
    
    if predictions_df.empty:
        print("No predictions found!")
        return
    
    print(f"Loaded {len(predictions_df)} predictions")
    
    # [2] 開発者ごとの的中率計算
    print("[2/6] Calculating developer accuracy...")
    developer_stats = calculate_developer_accuracy(predictions_df)
    
    print(f"Analyzed {len(developer_stats)} developers")
    print("\nSegment distribution:")
    print(developer_stats['segment'].value_counts().sort_index())
    
    # [3] 特徴量読み込み
    print("\n[3/6] Loading developer features...")
    developer_features = load_developer_features(data_path)
    
    print(f"Loaded features for {len(developer_features)} developers")
    print(f"Available features: {len(developer_features.columns)}")
    
    # [4] セグメント別特徴量分析
    print("\n[4/6] Analyzing segment characteristics...")
    segment_means, segment_diff, merged_df = analyze_segment_characteristics(
        developer_stats, developer_features
    )
    
    # 保存
    segment_means.to_csv(output_dir / 'segment_feature_means.csv')
    segment_diff.to_csv(output_dir / 'segment_feature_diff_standardized.csv')
    merged_df.to_csv(output_dir / 'developer_characteristics.csv')
    
    print(f"Saved segment analysis to: {output_dir}")
    
    # [5] ヒートマップ
    print("\n[5/6] Creating visualizations...")
    plot_segment_heatmap(segment_diff, vis_dir / 'developer_segments_heatmap.png')
    
    # PCA散布図
    plot_pca_scatter(merged_df, vis_dir / 'developer_pca_scatter.png')
    
    # [6] プロジェクト数の影響分析
    print("\n[6/6] Analyzing project count effect...")
    analyze_project_count_effect(
        merged_df,
        vis_dir / 'project_type_analysis.png'
    )
    
    print()
    print("="*80)
    print("Phase 3 Analysis Complete!")
    print("="*80)
    print(f"Results saved to: {output_dir}")
    print(f"Visualizations saved to: {vis_dir}")
    print()


if __name__ == '__main__':
    main()

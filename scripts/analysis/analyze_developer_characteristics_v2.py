#!/usr/bin/env python3
"""
Phase 3: 予測的中開発者の特性分析（完全版）

IRLモデルから抽出した14次元状態特徴量を使って、
開発者ごとの予測的中率を計算し、その特性を分析する。
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
    # 'activity_trend',  # 文字列なのでスキップ
    'collaboration_score',
    'code_quality_score',
    'recent_acceptance_rate',
    'review_load',
    # マルチプロジェクト特徴量
    'project_count',
    'project_activity_distribution',
    'main_project_contribution_ratio',
    'cross_project_collaboration_score'
]


def load_extracted_features(feature_path: Path) -> pd.DataFrame:
    """IRLモデルから抽出した特徴量を読み込み"""
    
    df = pd.read_csv(feature_path)
    
    # developer_idを設定
    df['developer_id'] = df['reviewer_email']
    
    # 予測を二値化
    df['pred_binary'] = (df['predicted_prob'] > 0.5).astype(int)
    
    # 正解判定
    df['correct'] = (df['pred_binary'] == df['true_label']).astype(int)
    
    return df


def calculate_developer_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """開発者ごとの予測的中率を計算"""
    
    developer_stats = df.groupby('developer_id').agg({
        'correct': ['sum', 'count', 'mean'],
        'predicted_prob': 'mean',
        'true_label': 'mean'
    })
    
    developer_stats.columns = ['correct_predictions', 'total_predictions', 'accuracy',
                                'avg_predicted_prob', 'actual_acceptance_rate']
    
    # セグメント分類
    developer_stats['segment'] = pd.cut(
        developer_stats['accuracy'],
        bins=[0, 0.5, 0.8, 0.95, 1.0, 1.01],
        labels=['Unpredictable', 'Low', 'Medium', 'High', 'Perfect'],
        include_lowest=True
    )
    
    return developer_stats


def analyze_segment_characteristics(
    developer_stats: pd.DataFrame,
    df_features: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """セグメント別の特徴量分析"""
    
    # 統合（developer_idをキーに）
    merged = developer_stats.join(df_features, how='inner')
    
    if merged.empty:
        print("No matching developers found")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # 利用可能な特徴量
    available_features = [f for f in STATE_FEATURES if f in merged.columns]
    
    # セグメント別の平均
    segment_means = merged.groupby('segment')[available_features].mean()
    
    # 全体平均
    overall_mean = merged[available_features].mean()
    
    # 差分（標準化）
    segment_std = merged[available_features].std()
    segment_diff = segment_means.sub(overall_mean, axis=1).div(segment_std, axis=1)
    
    return segment_means, segment_diff, merged


def plot_segment_heatmap(segment_diff: pd.DataFrame, output_path: Path):
    """セグメント別特徴量のヒートマップ"""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    sns.heatmap(
        segment_diff.T,
        cmap='RdBu_r',
        center=0,
        annot=True,
        fmt='.2f',
        cbar_kws={'label': 'Standardized Difference from Mean'},
        ax=ax,
        vmin=-2,
        vmax=2
    )
    
    ax.set_title('Developer Segment Characteristics\n(14-dim State Features from IRL Model)', 
                 fontsize=16, pad=20)
    ax.set_xlabel('Prediction Accuracy Segment', fontsize=12)
    ax.set_ylabel('State Feature (14 dimensions)', fontsize=12)
    
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
        s=150,
        alpha=0.7,
        edgecolors='black',
        linewidth=0.8
    )
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Prediction Accuracy', fontsize=13)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=13)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=13)
    ax.set_title('Developer Distribution in 14-dim State Feature Space (PCA)', 
                 fontsize=16, pad=20)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved PCA scatter plot: {output_path}")


def analyze_project_count_effect(merged_df: pd.DataFrame, output_dir: Path, vis_dir: Path):
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
        'project_count': 'mean',
        'cross_project_collaboration_score': 'mean'
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
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    # カウント別散布図
    ax = axes[1]
    scatter = ax.scatter(
        merged_df['project_count'],
        merged_df['accuracy'],
        alpha=0.6,
        s=120,
        c=merged_df['cross_project_collaboration_score'],
        cmap='coolwarm',
        edgecolors='black',
        linewidth=0.5
    )
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cross-Project Collaboration Score', fontsize=10)
    
    # トレンドライン
    z = np.polyfit(merged_df['project_count'], merged_df['accuracy'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(merged_df['project_count'].min(), 
                          merged_df['project_count'].max(), 100)
    ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, 
           label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')
    
    ax.set_title('Accuracy vs Project Count', fontsize=14)
    ax.set_xlabel('Number of Projects', fontsize=12)
    ax.set_ylabel('Prediction Accuracy', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(vis_dir / 'project_type_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved project type analysis: {vis_dir / 'project_type_analysis.png'}")
    
    # 統計をCSVに保存
    csv_path = output_dir / 'project_type_stats.csv'
    project_type_stats.to_csv(csv_path)
    print(f"Saved project type stats: {csv_path}")


def main():
    print("="*80)
    print("Phase 3: Developer Characteristics Analysis (Complete Version)")
    print("="*80)
    print()
    
    # 抽出済み特徴量を読み込み
    feature_path = ROOT / 'outputs' / 'analysis_data' / 'developer_state_features_0-3m.csv'
    
    if not feature_path.exists():
        print(f"Error: Feature file not found: {feature_path}")
        print("Please run extract_state_features.py first")
        return
    
    # 出力ディレクトリ
    output_dir = ROOT / 'outputs' / 'analysis_data'
    vis_dir = ROOT / 'outputs' / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # [1] 特徴量読み込み
    print("[1/6] Loading extracted features...")
    df = load_extracted_features(feature_path)
    
    print(f"Loaded {len(df)} predictions with 14-dim state features")
    print(f"Columns: {df.columns.tolist()}")
    
    # [2] 開発者ごとの的中率計算
    print("\n[2/6] Calculating developer accuracy...")
    developer_stats = calculate_developer_accuracy(df)
    
    print(f"Analyzed {len(developer_stats)} developers")
    print("\nSegment distribution:")
    print(developer_stats['segment'].value_counts().sort_index())
    
    # [3] 特徴量準備
    print("\n[3/6] Preparing features...")
    # developer_idをキーに特徴量を集約
    df_features = df.groupby('developer_id')[STATE_FEATURES].mean()
    
    print(f"Features for {len(df_features)} developers")
    print(f"Available features ({len(STATE_FEATURES)}): {STATE_FEATURES}")
    
    # [4] セグメント別特徴量分析
    print("\n[4/6] Analyzing segment characteristics...")
    segment_means, segment_diff, merged_df = analyze_segment_characteristics(
        developer_stats, df_features
    )
    
    if not merged_df.empty:
        # 保存
        segment_means.to_csv(output_dir / 'segment_feature_means.csv')
        segment_diff.to_csv(output_dir / 'segment_feature_diff_standardized.csv')
        merged_df.to_csv(output_dir / 'developer_characteristics_complete.csv')
        
        print(f"Saved segment analysis to: {output_dir}")
        
        # [5] ヒートマップ
        print("\n[5/6] Creating visualizations...")
        plot_segment_heatmap(segment_diff, vis_dir / 'developer_segments_heatmap.png')
        
        # PCA散布図
        plot_pca_scatter(merged_df, vis_dir / 'developer_pca_scatter.png')
        
        # [6] プロジェクト数の影響分析
        print("\n[6/6] Analyzing project count effect...")
        analyze_project_count_effect(merged_df, output_dir, vis_dir)
    
    print()
    print("="*80)
    print("Phase 3 Analysis Complete!")
    print("="*80)
    print(f"Results saved to: {output_dir}")
    print(f"Visualizations saved to: {vis_dir}")
    print()


if __name__ == '__main__':
    main()

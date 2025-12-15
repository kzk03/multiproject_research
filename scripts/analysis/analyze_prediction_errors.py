#!/usr/bin/env python3
"""
Phase 4: 誤予測パターンの深堀り分析

False Positive（承諾すると予測したが拒否）と
False Negative（拒否すると予測したが承諾）のパターンを分析する。
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# パス設定
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")


def load_predictions_with_features(
    model_dir: Path,
    data_path: Path
) -> pd.DataFrame:
    """予測結果と特徴量を統合"""
    
    # 予測結果読み込み
    predictions = []
    for train_period in ['0-3m', '3-6m', '6-9m', '9-12m']:
        for eval_period in ['0-3m', '3-6m', '6-9m', '9-12m']:
            pred_file = model_dir / f'train_{train_period}' / f'eval_{eval_period}' / 'predictions.csv'
            
            if pred_file.exists():
                df = pd.read_csv(pred_file)
                df['train_period'] = train_period
                df['eval_period'] = eval_period
                predictions.append(df)
    
    if not predictions:
        return pd.DataFrame()
    
    pred_df = pd.concat(predictions, ignore_index=True)
    
    # 特徴量データ読み込み
    feat_df = pd.read_csv(data_path)
    
    # reviewer_emailをキーに結合
    if 'reviewer_email' in pred_df.columns and 'reviewer_email' in feat_df.columns:
        merged = pred_df.merge(
            feat_df,
            on='reviewer_email',
            how='left',
            suffixes=('', '_feat')
        )
    else:
        merged = pred_df
    
    return merged


def classify_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """予測を4分類（TP, TN, FP, FN）"""
    
    # 予測を二値化
    if 'predicted_binary' in df.columns:
        df['pred_binary'] = df['predicted_binary']
    elif 'predicted_prob' in df.columns:
        df['pred_binary'] = (df['predicted_prob'] > 0.5).astype(int)
        df['prediction'] = df['predicted_prob']  # 後続処理のため
    elif 'prediction' in df.columns:
        df['pred_binary'] = (df['prediction'] > 0.5).astype(int)
    elif 'predicted_label' in df.columns:
        df['pred_binary'] = df['predicted_label']
    else:
        print("Warning: No prediction column")
        print(f"Available columns: {df.columns.tolist()}")
        return df

    # 正解ラベル
    if 'true_label' in df.columns:
        label_col = 'true_label'
    elif 'label' in df.columns:
        label_col = 'label'
    else:
        print("Warning: No label column")
        print(f"Available columns: {df.columns.tolist()}")
        return df
    
    # 4分類
    def classify_row(row):
        pred = row['pred_binary']
        true = row[label_col]
        
        if pred == 1 and true == 1:
            return 'TP'  # True Positive
        elif pred == 0 and true == 0:
            return 'TN'  # True Negative
        elif pred == 1 and true == 0:
            return 'FP'  # False Positive（承諾すると予測したが拒否）
        else:  # pred == 0 and true == 1
            return 'FN'  # False Negative（拒否すると予測したが承諾）
    
    df['prediction_class'] = df.apply(classify_row, axis=1)
    
    return df


def analyze_false_positives(df_fp: pd.DataFrame, output_dir: Path):
    """False Positive（偽陽性）の詳細分析"""
    
    print("\n" + "="*80)
    print("False Positive Analysis")
    print("="*80)
    print(f"Total FP cases: {len(df_fp)}")
    
    if len(df_fp) == 0:
        print("No false positives found!")
        return
    
    # 予測スコアの分布
    if 'prediction' in df_fp.columns:
        print(f"\nPrediction score distribution:")
        print(df_fp['prediction'].describe())
        
        # 高スコアで誤った例（確信度が高いのに間違い）
        high_conf_fp = df_fp[df_fp['prediction'] > 0.8]
        print(f"\nHigh-confidence FP (score > 0.8): {len(high_conf_fp)} cases")
    
    # プロジェクト数の影響
    if 'project_count' in df_fp.columns:
        print(f"\nProject count distribution:")
        print(df_fp['project_count'].value_counts().sort_index())
    
    # クロスプロジェクト活動
    if 'is_cross_project' in df_fp.columns:
        cross_proj_ratio = df_fp['is_cross_project'].mean()
        print(f"\nCross-project activity ratio: {cross_proj_ratio:.1%}")
    
    # 活動トレンド
    if 'activity_trend' in df_fp.columns:
        print(f"\nActivity trend distribution:")
        print(df_fp['activity_trend'].value_counts())
    
    # CSV保存
    fp_summary = df_fp[[
        'reviewer_email',
        'prediction',
        'project_count',
        'recent_acceptance_rate',
        'review_load',
        'train_period',
        'eval_period'
    ]].copy() if all(c in df_fp.columns for c in [
        'reviewer_email', 'prediction', 'project_count',
        'recent_acceptance_rate', 'review_load'
    ]) else df_fp[['train_period', 'eval_period']].copy()
    
    fp_summary.to_csv(output_dir / 'false_positive_cases.csv', index=False)
    print(f"\nSaved FP cases to: {output_dir / 'false_positive_cases.csv'}")
    print("="*80)


def analyze_false_negatives(df_fn: pd.DataFrame, output_dir: Path):
    """False Negative（偽陰性）の詳細分析"""
    
    print("\n" + "="*80)
    print("False Negative Analysis")
    print("="*80)
    print(f"Total FN cases: {len(df_fn)}")
    
    if len(df_fn) == 0:
        print("No false negatives found!")
        return
    
    # 予測スコアの分布
    if 'prediction' in df_fn.columns:
        print(f"\nPrediction score distribution:")
        print(df_fn['prediction'].describe())
        
        # 低スコアで誤った例（確信度が高いのに間違い）
        low_conf_fn = df_fn[df_fn['prediction'] < 0.2]
        print(f"\nHigh-confidence FN (score < 0.2): {len(low_conf_fn)} cases")
    
    # 経験日数（新規参入者?）
    if 'experience_days' in df_fn.columns:
        print(f"\nExperience days distribution:")
        print(df_fn['experience_days'].describe())
        
        new_devs = df_fn[df_fn['experience_days'] < 90]
        print(f"New developers (<90 days): {len(new_devs)} cases ({len(new_devs)/len(df_fn):.1%})")
    
    # プロジェクト数
    if 'project_count' in df_fn.columns:
        print(f"\nProject count distribution:")
        print(df_fn['project_count'].value_counts().sort_index())
    
    # CSV保存
    fn_summary = df_fn[[
        'reviewer_email',
        'prediction',
        'experience_days',
        'project_count',
        'recent_acceptance_rate',
        'train_period',
        'eval_period'
    ]].copy() if all(c in df_fn.columns for c in [
        'reviewer_email', 'prediction', 'experience_days',
        'project_count', 'recent_acceptance_rate'
    ]) else df_fn[['train_period', 'eval_period']].copy()
    
    fn_summary.to_csv(output_dir / 'false_negative_cases.csv', index=False)
    print(f"\nSaved FN cases to: {output_dir / 'false_negative_cases.csv'}")
    print("="*80)


def plot_error_distribution(df: pd.DataFrame, output_path: Path):
    """エラー分布の可視化"""
    
    # 分類カウント
    class_counts = df['prediction_class'].value_counts()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # [1] 混同行列スタイルのヒートマップ
    ax = axes[0, 0]
    confusion = pd.crosstab(
        df['pred_binary'],
        df['label'] if 'label' in df.columns else df['true_label'],
        normalize='all'
    ) * 100
    
    sns.heatmap(
        confusion,
        annot=True,
        fmt='.1f',
        cmap='Blues',
        cbar_kws={'label': 'Percentage (%)'},
        ax=ax
    )
    ax.set_title('Confusion Matrix (Normalized)', fontsize=14)
    ax.set_xlabel('True Label', fontsize=12)
    ax.set_ylabel('Predicted Label', fontsize=12)
    
    # [2] 4分類の割合
    ax = axes[0, 1]
    colors = {'TP': '#2ecc71', 'TN': '#3498db', 'FP': '#e74c3c', 'FN': '#f39c12'}
    class_colors = [colors.get(c, '#95a5a6') for c in class_counts.index]
    
    wedges, texts, autotexts = ax.pie(
        class_counts.values,
        labels=class_counts.index,
        autopct='%1.1f%%',
        colors=class_colors,
        startangle=90
    )
    ax.set_title('Prediction Classification Distribution', fontsize=14)
    
    # [3] 予測スコア分布（FP vs FN）
    ax = axes[1, 0]
    if 'prediction' in df.columns:
        df_fp = df[df['prediction_class'] == 'FP']
        df_fn = df[df['prediction_class'] == 'FN']
        
        ax.hist(df_fp['prediction'], bins=20, alpha=0.6, label='FP', color='#e74c3c')
        ax.hist(df_fn['prediction'], bins=20, alpha=0.6, label='FN', color='#f39c12')
        
        ax.set_xlabel('Prediction Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Prediction Score Distribution (Errors Only)', fontsize=14)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No prediction scores available',
               ha='center', va='center', fontsize=12)
        ax.axis('off')
    
    # [4] 期間別エラー率
    ax = axes[1, 1]
    if 'eval_period' in df.columns:
        period_error = df.groupby('eval_period')['prediction_class'].apply(
            lambda x: (x.isin(['FP', 'FN']).sum() / len(x)) * 100
        ).sort_index()
        
        ax.bar(range(len(period_error)), period_error.values, color='#e74c3c', alpha=0.7)
        ax.set_xticks(range(len(period_error)))
        ax.set_xticklabels(period_error.index, rotation=45)
        ax.set_xlabel('Evaluation Period', fontsize=12)
        ax.set_ylabel('Error Rate (%)', fontsize=12)
        ax.set_title('Error Rate by Evaluation Period', fontsize=14)
        ax.grid(axis='y', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No period information available',
               ha='center', va='center', fontsize=12)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved error distribution plot: {output_path}")


def main():
    print("="*80)
    print("Phase 4: Prediction Error Analysis")
    print("="*80)
    print()
    
    # モデル選択
    model_dir = ROOT / 'outputs' / '50projects_irl' / 'no_os'
    data_path = ROOT / 'data' / 'openstack_50proj_2021_2024_feat.csv'
    
    # 出力ディレクトリ
    output_dir = ROOT / 'outputs' / 'analysis_data'
    vis_dir = ROOT / 'outputs' / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # [1] データ読み込み
    print("[1/4] Loading predictions and features...")
    df = load_predictions_with_features(model_dir, data_path)
    
    if df.empty:
        print("No data found!")
        return
    
    print(f"Loaded {len(df)} predictions")
    
    # [2] 予測分類
    print("[2/4] Classifying predictions...")
    df = classify_predictions(df)
    
    print("\nPrediction classification:")
    print(df['prediction_class'].value_counts())
    
    # [3] エラー分析
    print("\n[3/4] Analyzing errors...")
    
    df_fp = df[df['prediction_class'] == 'FP']
    df_fn = df[df['prediction_class'] == 'FN']
    
    analyze_false_positives(df_fp, output_dir)
    analyze_false_negatives(df_fn, output_dir)
    
    # [4] 可視化
    print("\n[4/4] Creating visualizations...")
    plot_error_distribution(df, vis_dir / 'error_analysis.png')
    
    # 全エラーケースを保存
    df_errors = df[df['prediction_class'].isin(['FP', 'FN'])]
    df_errors.to_csv(output_dir / 'all_error_cases.csv', index=False)
    print(f"Saved all error cases: {output_dir / 'all_error_cases.csv'}")
    
    print()
    print("="*80)
    print("Phase 4 Analysis Complete!")
    print("="*80)
    print(f"Results saved to: {output_dir}")
    print(f"Visualizations saved to: {vis_dir}")
    print()


if __name__ == '__main__':
    main()

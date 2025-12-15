#!/usr/bin/env python3
"""
Random Forestを全10パターンで評価

IRL時系列版と同じ10パターンでRFを訓練・評価して公平な比較を行う。
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_features(df: pd.DataFrame):
    """特徴量を準備（19次元）"""
    state_features = [
        'experience_days',
        'total_changes',
        'total_reviews',
        'recent_activity_frequency',
        'avg_activity_gap',
        'activity_trend',
        'collaboration_score',
        'code_quality_score',
        'recent_acceptance_rate',
        'review_load',
        'project_count',
        'project_activity_distribution',
        'main_project_contribution_ratio',
        'cross_project_collaboration_score',
    ]

    action_features = [
        'avg_action_intensity',
        'avg_collaboration',
        'avg_response_time',
        'avg_review_size',
        'cross_project_action_ratio',
    ]

    all_features = state_features + action_features

    # activity_trendを数値に変換
    trend_mapping = {
        'increasing': 1.0,
        'stable': 0.0,
        'decreasing': -1.0
    }
    if 'activity_trend' in df.columns:
        df['activity_trend'] = df['activity_trend'].map(trend_mapping).fillna(0)

    X = df[all_features].fillna(0)
    y = df['true_label']

    return X, y, all_features


def train_and_evaluate_rf(X_train, y_train, X_test, y_test, pattern_name: str):
    """Random Forestを訓練・評価"""
    logger.info(f"パターン {pattern_name}: Random Forest訓練中...")
    
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    start_time = time.time()
    rf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    start_time = time.time()
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    y_pred = rf.predict(X_test)
    predict_time = time.time() - start_time
    
    # メトリクス計算
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    try:
        auc_roc = roc_auc_score(y_test, y_pred_proba)
    except:
        auc_roc = 0.5
    
    try:
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
        auc_pr = auc(recall_vals, precision_vals)
    except:
        auc_pr = 0.0
    
    results = {
        'pattern': pattern_name,
        'f1': f1,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'train_time': train_time,
        'predict_time': predict_time,
        'n_samples': len(y_test),
        'n_positive': int(y_test.sum()),
        'n_negative': int(len(y_test) - y_test.sum())
    }
    
    logger.info(f"  F1={f1:.4f}, AUC-ROC={auc_roc:.4f}, Accuracy={accuracy:.4f}, Train={train_time:.4f}s")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Random Forestを全10パターンで評価')
    parser.add_argument('--irl-output', required=True, help='IRL時系列版の出力ディレクトリ')
    parser.add_argument('--output', required=True, help='RF評価結果の出力ディレクトリ')
    
    args = parser.parse_args()
    
    irl_output_dir = Path(args.irl_output)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 10パターンのパスを特定
    patterns = [
        ('train_0-3m', 'eval_0-3m'),
        ('train_0-3m', 'eval_3-6m'),
        ('train_0-3m', 'eval_6-9m'),
        ('train_0-3m', 'eval_9-12m'),
        ('train_3-6m', 'eval_3-6m'),
        ('train_3-6m', 'eval_6-9m'),
        ('train_3-6m', 'eval_9-12m'),
        ('train_6-9m', 'eval_6-9m'),
        ('train_6-9m', 'eval_9-12m'),
        ('train_9-12m', 'eval_9-12m'),
    ]
    
    all_results = []
    
    for train_dir, eval_dir in patterns:
        pattern_name = f"{train_dir.replace('train_', '')} → {eval_dir.replace('eval_', '')}"
        
        # 訓練データの特徴量を読み込み
        train_features_path = irl_output_dir / train_dir / 'developer_state_features.csv'
        eval_features_path = irl_output_dir / train_dir / eval_dir / 'developer_state_features.csv'
        
        if not train_features_path.exists():
            logger.warning(f"訓練データが見つかりません: {train_features_path}")
            continue
        
        if not eval_features_path.exists():
            logger.warning(f"評価データが見つかりません: {eval_features_path}")
            continue
        
        logger.info(f"\n{'='*80}")
        logger.info(f"パターン: {pattern_name}")
        logger.info(f"{'='*80}")
        
        # データ読み込み
        train_df = pd.read_csv(train_features_path)
        eval_df = pd.read_csv(eval_features_path)
        
        # 特徴量準備
        X_train, y_train, feature_names = prepare_features(train_df)
        X_eval, y_eval, _ = prepare_features(eval_df)
        
        logger.info(f"訓練サンプル: {len(X_train)}, 評価サンプル: {len(X_eval)}")
        logger.info(f"訓練正例率: {y_train.mean()*100:.1f}%, 評価正例率: {y_eval.mean()*100:.1f}%")
        
        # Random Forest訓練・評価
        results = train_and_evaluate_rf(
            X_train.values, y_train.values,
            X_eval.values, y_eval.values,
            pattern_name
        )
        
        all_results.append(results)
    
    # 結果をCSVに保存
    results_df = pd.DataFrame(all_results)
    results_csv = output_dir / 'rf_all_patterns_results.csv'
    results_df.to_csv(results_csv, index=False)
    logger.info(f"\n結果を保存: {results_csv}")
    
    # サマリー出力
    logger.info("\n" + "="*80)
    logger.info("Random Forest 全パターン評価結果")
    logger.info("="*80)
    logger.info(f"\n平均F1スコア: {results_df['f1'].mean():.4f}")
    logger.info(f"最高F1スコア: {results_df['f1'].max():.4f} ({results_df.loc[results_df['f1'].idxmax(), 'pattern']})")
    logger.info(f"最低F1スコア: {results_df['f1'].min():.4f} ({results_df.loc[results_df['f1'].idxmin(), 'pattern']})")
    logger.info(f"\n平均訓練時間: {results_df['train_time'].mean():.4f}秒")
    logger.info(f"総訓練時間: {results_df['train_time'].sum():.4f}秒")
    
    # F1スコア行列を作成
    matrix = np.full((4, 4), np.nan)
    train_windows = ['0-3m', '3-6m', '6-9m', '9-12m']
    
    for _, row in results_df.iterrows():
        pattern = row['pattern']
        train_win, eval_win = pattern.split(' → ')
        i = train_windows.index(train_win)
        j = train_windows.index(eval_win)
        matrix[i, j] = row['f1']
    
    matrix_df = pd.DataFrame(matrix, index=train_windows, columns=train_windows)
    matrix_csv = output_dir / 'rf_f1_matrix.csv'
    matrix_df.to_csv(matrix_csv)
    logger.info(f"\nF1スコア行列を保存: {matrix_csv}")
    logger.info(f"\n{matrix_df}")
    
    logger.info("\n" + "="*80)
    logger.info("すべての評価が完了しました！")
    logger.info("="*80)


if __name__ == '__main__':
    main()

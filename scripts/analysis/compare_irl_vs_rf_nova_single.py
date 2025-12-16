#!/usr/bin/env python3
"""
IRL vs Random Forest 比較 - Nova単体プロジェクト用

Nova単体プロジェクトの結果でRF比較を実施
特徴量: 状態10次元、行動4次元（マルチプロジェクト特徴量を除外）

重要: IRLと同じ訓練/評価分割を使用
- 訓練: 2021年のデータ
- 評価: 2023年のデータ（時系列分割）
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
    """
    Nova単体用の特徴量を準備

    状態特徴量: 10次元（マルチプロジェクト特徴量4つを除外）
    行動特徴量: 4次元（cross_project_action_ratioを除外）
    """
    # 状態特徴量（10次元）
    # マルチプロジェクト特徴量を除外:
    # - project_count
    # - project_activity_distribution
    # - main_project_contribution_ratio
    # - cross_project_collaboration_score
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
        # 以下はマルチプロジェクト特徴量なので除外
        # 'project_count',
        # 'project_activity_distribution',
        # 'main_project_contribution_ratio',
        # 'cross_project_collaboration_score',
    ]

    # 行動特徴量（4次元）
    # cross_project_action_ratioを除外
    action_features = [
        'avg_action_intensity',
        'avg_collaboration',
        'avg_response_time',
        'avg_review_size',
        # 以下はマルチプロジェクト特徴量なので除外
        # 'cross_project_action_ratio',
    ]

    all_features = state_features + action_features

    logger.info(f"使用特徴量: {len(all_features)}次元")
    logger.info(f"  状態特徴量: {len(state_features)}次元")
    logger.info(f"  行動特徴量: {len(action_features)}次元")

    # activity_trend変換
    trend_mapping = {'increasing': 1.0, 'stable': 0.0, 'decreasing': -1.0}
    if 'activity_trend' in df.columns:
        df['activity_trend'] = df['activity_trend'].map(trend_mapping).fillna(0)

    X = df[all_features].fillna(0)
    y = df['true_label']

    return X, y, all_features


def main():
    import argparse

    parser = argparse.ArgumentParser(description='IRL vs RF 比較 - Nova単体用')
    parser.add_argument('--train-features', required=True, help='訓練特徴量CSV')
    parser.add_argument('--eval-features', required=True, help='評価特徴量CSV')
    parser.add_argument('--output', required=True, help='出力ディレクトリ')

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Nova単体プロジェクト: IRL vs Random Forest 比較")
    logger.info("特徴量: 状態10次元 + 行動4次元（マルチプロジェクト特徴量除外）")
    logger.info("=" * 80)

    # 訓練データ読み込み
    logger.info(f"\n訓練データ読み込み: {args.train_features}")
    train_df = pd.read_csv(args.train_features)
    X_train, y_train, feature_names = prepare_features(train_df)

    logger.info(f"訓練サンプル: {len(X_train)}")
    logger.info(f"訓練正例: {y_train.sum()} / {len(y_train)} ({y_train.mean()*100:.1f}%)")

    # 評価データ読み込み
    logger.info(f"\n評価データ読み込み: {args.eval_features}")
    eval_df = pd.read_csv(args.eval_features)
    X_eval, y_eval, _ = prepare_features(eval_df)

    logger.info(f"評価サンプル: {len(X_eval)}")
    logger.info(f"評価正例: {y_eval.sum()} / {len(y_eval)} ({y_eval.mean()*100:.1f}%)")

    # Random Forest訓練
    logger.info("\n" + "=" * 80)
    logger.info("Random Forest訓練（訓練データのみ使用）")
    logger.info("=" * 80)

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
    rf.fit(X_train.values, y_train.values)
    train_time = time.time() - start_time

    logger.info(f"訓練完了: {train_time:.4f}秒")

    # 評価データで予測
    logger.info("\n" + "=" * 80)
    logger.info("評価データで予測（データリークなし）")
    logger.info("=" * 80)

    start_time = time.time()
    y_pred_proba = rf.predict_proba(X_eval.values)[:, 1]
    y_pred = rf.predict(X_eval.values)
    predict_time = time.time() - start_time

    # メトリクス計算
    f1 = f1_score(y_eval, y_pred)
    precision = precision_score(y_eval, y_pred, zero_division=0)
    recall = recall_score(y_eval, y_pred)
    accuracy = accuracy_score(y_eval, y_pred)

    try:
        auc_roc = roc_auc_score(y_eval, y_pred_proba)
    except:
        auc_roc = 0.5

    try:
        precision_vals, recall_vals, _ = precision_recall_curve(y_eval, y_pred_proba)
        auc_pr = auc(recall_vals, precision_vals)
    except:
        auc_pr = 0.0

    # 混同行列
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_eval, y_pred).ravel()

    # 結果出力
    logger.info("\n" + "=" * 80)
    logger.info("Random Forest 評価結果（Nova単体、状態10+行動4）")
    logger.info("=" * 80)
    logger.info(f"F1 Score:   {f1:.4f}")
    logger.info(f"AUC-ROC:    {auc_roc:.4f}")
    logger.info(f"AUC-PR:     {auc_pr:.4f}")
    logger.info(f"Precision:  {precision:.4f}")
    logger.info(f"Recall:     {recall:.4f}")
    logger.info(f"Accuracy:   {accuracy:.4f}")
    logger.info(f"\n混同行列:")
    logger.info(f"  TP: {tp:3d}  FN: {fn:3d}")
    logger.info(f"  FP: {fp:3d}  TN: {tn:3d}")
    logger.info(f"\n訓練時間: {train_time:.4f}秒")
    logger.info(f"予測時間: {predict_time:.4f}秒")

    # 特徴量重要度
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    logger.info("\n特徴量重要度 Top 10:")
    for i, row in feature_importance.head(10).iterrows():
        logger.info(f"  {row['feature']:30s}: {row['importance']:.4f}")

    # 結果保存
    results = {
        'model': 'Random Forest (Nova Single Project)',
        'feature_config': {
            'state_dim': 10,
            'action_dim': 4,
            'total_features': 14,
            'excluded_features': [
                'project_count',
                'project_activity_distribution',
                'main_project_contribution_ratio',
                'cross_project_collaboration_score',
                'cross_project_action_ratio'
            ]
        },
        'f1': float(f1),
        'auc_roc': float(auc_roc),
        'auc_pr': float(auc_pr),
        'precision': float(precision),
        'recall': float(recall),
        'accuracy': float(accuracy),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'train_time': float(train_time),
        'predict_time': float(predict_time),
        'train_samples': len(X_train),
        'eval_samples': len(X_eval)
    }

    results_path = output_dir / 'rf_nova_single_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # 特徴量重要度保存
    importance_path = output_dir / 'rf_nova_single_feature_importance.csv'
    feature_importance.to_csv(importance_path, index=False)

    logger.info(f"\n結果を保存: {results_path}")
    logger.info(f"特徴量重要度を保存: {importance_path}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

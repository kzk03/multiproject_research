#!/usr/bin/env python3
"""
Random Forest 10パターン評価（生データから）

50projects_irl_timeseries/2x_osと同じ10パターンでRFを評価するため、
生データから特徴量を抽出してRF訓練・評価を行う。
"""

import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, auc, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, confusion_matrix
)

# 既存のIRL predictorから特徴量抽出関数を import
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from review_predictor.model.irl_predictor import (
    extract_developer_state, extract_developer_action,
    DeveloperState, DeveloperAction
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data():
    """50プロジェクトデータを読み込み"""
    data_path = Path("/Users/kazuki-h/research/multiproject_research/data/openstack_50proj_2021_2024_feat.csv")
    logger.info(f"データ読み込み: {data_path}")
    df = pd.read_csv(data_path)
    df['created'] = pd.to_datetime(df['created'])
    logger.info(f"  総レコード数: {len(df)}")
    return df


def extract_features_for_period(df: pd.DataFrame, train_start: str, train_end: str,
                                 eval_start: str, eval_end: str):
    """
    指定期間の特徴量を抽出

    Args:
        df: 元データ
        train_start, train_end: 訓練期間（観測期間）
        eval_start, eval_end: 評価期間（予測ターゲット）

    Returns:
        features_df: 特徴量DataFrame
    """
    train_start_dt = pd.to_datetime(train_start)
    train_end_dt = pd.to_datetime(train_end)
    eval_start_dt = pd.to_datetime(eval_start)
    eval_end_dt = pd.to_datetime(eval_end)

    # 訓練期間のデータ
    train_mask = (df['created'] >= train_start_dt) & (df['created'] < train_end_dt)
    train_data = df[train_mask]

    # 評価期間のデータ
    eval_mask = (df['created'] >= eval_start_dt) & (df['created'] < eval_end_dt)
    eval_data = df[eval_mask]

    logger.info(f"  訓練期間 {train_start} to {train_end}: {len(train_data)} records")
    logger.info(f"  評価期間 {eval_start} to {eval_end}: {len(eval_data)} records")

    # 訓練期間に活動していた開発者
    developers = train_data['email'].unique()
    logger.info(f"  訓練期間の開発者数: {len(developers)}")

    features_list = []

    for dev_email in developers:
        # 訓練期間のデータで状態・行動を抽出
        dev_train_data = train_data[train_data['email'] == dev_email]

        try:
            # 状態抽出
            state = extract_developer_state(
                dev_train_data,
                dev_email,
                eval_start_dt
            )

            # 行動抽出
            action = extract_developer_action(
                dev_train_data,
                dev_email
            )

            # 評価期間のラベル（継続=1, 離脱=0）
            dev_eval_data = eval_data[eval_data['email'] == dev_email]
            true_label = 1 if len(dev_eval_data) > 0 else 0

            # 特徴量を辞書化
            features = {
                'email': dev_email,
                # 状態特徴量（14次元）
                'experience_days': state.experience_days,
                'total_changes': state.total_changes,
                'total_reviews': state.total_reviews,
                'recent_activity_frequency': state.recent_activity_frequency,
                'avg_activity_gap': state.avg_activity_gap,
                'activity_trend': state.activity_trend,
                'collaboration_score': state.collaboration_score,
                'code_quality_score': state.code_quality_score,
                'recent_acceptance_rate': state.recent_acceptance_rate,
                'review_load': state.review_load,
                'project_count': state.project_count,
                'project_activity_distribution': state.project_activity_distribution,
                'main_project_contribution_ratio': state.main_project_contribution_ratio,
                'cross_project_collaboration_score': state.cross_project_collaboration_score,
                # 行動特徴量（5次元）
                'avg_action_intensity': action.avg_action_intensity,
                'avg_collaboration': action.avg_collaboration,
                'avg_response_time': action.avg_response_time,
                'avg_review_size': action.avg_review_size,
                'cross_project_action_ratio': action.cross_project_action_ratio,
                # ラベル
                'true_label': true_label
            }
            features_list.append(features)

        except Exception as e:
            logger.warning(f"  開発者 {dev_email} の特徴量抽出失敗: {e}")
            continue

    features_df = pd.DataFrame(features_list)
    logger.info(f"  特徴量抽出完了: {len(features_df)} developers")
    logger.info(f"  継続率: {features_df['true_label'].mean()*100:.1f}%")

    return features_df


def prepare_features_for_rf(df: pd.DataFrame):
    """RFの入力用に特徴量を準備"""
    state_features = [
        'experience_days', 'total_changes', 'total_reviews',
        'recent_activity_frequency', 'avg_activity_gap', 'activity_trend',
        'collaboration_score', 'code_quality_score', 'recent_acceptance_rate',
        'review_load', 'project_count', 'project_activity_distribution',
        'main_project_contribution_ratio', 'cross_project_collaboration_score',
    ]

    action_features = [
        'avg_action_intensity', 'avg_collaboration', 'avg_response_time',
        'avg_review_size', 'cross_project_action_ratio',
    ]

    all_features = state_features + action_features

    # activity_trend変換
    trend_mapping = {'increasing': 1.0, 'stable': 0.0, 'decreasing': -1.0}
    df = df.copy()
    if 'activity_trend' in df.columns:
        df['activity_trend'] = df['activity_trend'].map(trend_mapping).fillna(0)

    X = df[all_features].fillna(0)
    y = df['true_label']

    return X, y


def train_and_evaluate_rf(X_train, y_train, X_eval, y_eval):
    """Random Forestを訓練・評価"""
    start_time = time.time()

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )

    rf.fit(X_train, y_train)
    train_time = time.time() - start_time

    # 予測
    start_time = time.time()
    y_pred_proba = rf.predict_proba(X_eval)[:, 1]
    y_pred = rf.predict(X_eval)
    predict_time = time.time() - start_time

    # メトリクス
    f1 = f1_score(y_eval, y_pred)
    auc_roc = roc_auc_score(y_eval, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_eval, y_pred_proba)
    auc_pr = auc(recall, precision)
    prec = precision_score(y_eval, y_pred, zero_division=0)
    rec = recall_score(y_eval, y_pred, zero_division=0)
    acc = accuracy_score(y_eval, y_pred)

    # 混同行列
    tn, fp, fn, tp = confusion_matrix(y_eval, y_pred).ravel()

    return {
        'f1': f1,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'precision': prec,
        'recall': rec,
        'accuracy': acc,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'train_time': train_time,
        'predict_time': predict_time
    }


def main():
    logger.info("=" * 80)
    logger.info("Random Forest 10パターン評価（生データから）")
    logger.info("=" * 80)

    # データ読み込み
    df = load_data()

    # 10パターン定義（2x OSバージョンに合わせる）
    patterns = [
        ("0-3m", "2021-01-01", "2021-04-01", "2021-01-01", "2021-04-01"),
        ("0-3m → 3-6m", "2021-01-01", "2021-04-01", "2021-04-01", "2021-07-01"),
        ("0-3m → 6-9m", "2021-01-01", "2021-04-01", "2021-07-01", "2021-10-01"),
        ("0-3m → 9-12m", "2021-01-01", "2021-04-01", "2021-10-01", "2022-01-01"),
        ("3-6m", "2021-04-01", "2021-07-01", "2021-04-01", "2021-07-01"),
        ("3-6m → 6-9m", "2021-04-01", "2021-07-01", "2021-07-01", "2021-10-01"),
        ("3-6m → 9-12m", "2021-04-01", "2021-07-01", "2021-10-01", "2022-01-01"),
        ("6-9m", "2021-07-01", "2021-10-01", "2021-07-01", "2021-10-01"),
        ("6-9m → 9-12m", "2021-07-01", "2021-10-01", "2021-10-01", "2022-01-01"),
        ("9-12m", "2021-10-01", "2022-01-01", "2021-10-01", "2022-01-01")
    ]

    all_results = []

    for pattern_name, train_start, train_end, eval_start, eval_end in patterns:
        logger.info(f"\n{'='*80}")
        logger.info(f"パターン: {pattern_name}")
        logger.info(f"{'='*80}")

        # 訓練データの特徴量抽出
        logger.info("訓練データ特徴量抽出中...")
        train_features = extract_features_for_period(
            df, train_start, train_end, eval_start, eval_end
        )

        X_train, y_train = prepare_features_for_rf(train_features)

        logger.info(f"\n訓練サンプル: {len(X_train)} (Positive: {y_train.sum()}, Negative: {len(y_train)-y_train.sum()})")

        # Random Forest訓練・評価
        logger.info("Random Forest訓練・評価中...")
        results = train_and_evaluate_rf(X_train, y_train, X_train, y_train)  # まず訓練データで

        results['pattern'] = pattern_name
        results['train_window'] = pattern_name.split(' → ')[0] if ' → ' in pattern_name else pattern_name
        results['eval_window'] = pattern_name.split(' → ')[1] if ' → ' in pattern_name else pattern_name
        results['n_samples'] = len(X_train)

        logger.info(f"\n結果:")
        logger.info(f"  F1:        {results['f1']:.4f}")
        logger.info(f"  AUC-ROC:   {results['auc_roc']:.4f}")
        logger.info(f"  AUC-PR:    {results['auc_pr']:.4f}")
        logger.info(f"  Recall:    {results['recall']:.4f}")
        logger.info(f"  Precision: {results['precision']:.4f}")
        logger.info(f"  Accuracy:  {results['accuracy']:.4f}")
        logger.info(f"  TP={results['tp']}, TN={results['tn']}, FP={results['fp']}, FN={results['fn']}")

        all_results.append(results)

    # 結果保存
    output_dir = Path("/Users/kazuki-h/research/multiproject_research/outputs/analysis_data/rf_10patterns")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(all_results)

    # CSV保存
    csv_path = output_dir / "rf_10patterns_from_raw.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"\n✓ 結果をCSVに保存: {csv_path}")

    # JSON保存
    json_path = output_dir / "rf_10patterns_from_raw.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"✓ 結果をJSONに保存: {json_path}")

    # サマリー
    logger.info("\n" + "=" * 80)
    logger.info("サマリー")
    logger.info("=" * 80)
    logger.info(f"\n平均メトリクス（{len(all_results)}パターン）:")
    logger.info(f"  平均 F1:        {results_df['f1'].mean():.4f}")
    logger.info(f"  平均 AUC-ROC:   {results_df['auc_roc'].mean():.4f}")
    logger.info(f"  平均 Recall:    {results_df['recall'].mean():.4f}")
    logger.info(f"  平均 Precision: {results_df['precision'].mean():.4f}")


if __name__ == "__main__":
    main()

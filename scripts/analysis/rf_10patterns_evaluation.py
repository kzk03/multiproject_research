#!/usr/bin/env python3
"""
Random Forest 10パターン評価（Nova単体 & マルチプロジェクト）

IRLと同じ10パターンでRFを評価して公平な比較を実現
"""

import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(project_type='multi'):
    """データを読み込み"""
    base_path = Path("/Users/kazuki-h/research/multiproject_research/data")

    if project_type == 'multi':
        # マルチプロジェクト（50プロジェクト）
        data_path = base_path / "openstack_50proj_2021_2024_feat.csv"
        logger.info(f"マルチプロジェクトデータ読み込み: {data_path}")
    else:
        # Nova単体
        data_path = base_path / "openstack_50proj_2021_2024_feat.csv"
        logger.info(f"Nova単体データ読み込み: {data_path}")

    df = pd.read_csv(data_path)

    # タイムスタンプカラムを確認して使用
    if 'created' in df.columns:
        df['timestamp'] = pd.to_datetime(df['created'])
    elif 'request_time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['request_time'])
    elif 'context_date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['context_date'])
    else:
        raise ValueError("No timestamp column found")

    # emailカラムを確認
    if 'email' not in df.columns:
        if 'developer_email' in df.columns:
            df['email'] = df['developer_email']
        elif 'reviewer_email' in df.columns:
            df['email'] = df['reviewer_email']
        elif 'owner_email' in df.columns:
            df['email'] = df['owner_email']

    # Nova単体の場合はフィルタ
    if project_type == 'nova':
        df = df[df['project'] == 'openstack/nova']
        logger.info(f"  Nova単体にフィルタ: {len(df)} records")
    else:
        logger.info(f"  総レコード数: {len(df)}")

    return df


def extract_features_for_window(
    df,
    feature_start,
    feature_end,
    label_start,
    label_end,
    project_type='multi',
    require_label_activity: bool = False,
):
    """
    指定期間の開発者を抽出し、特徴量を計算

    簡易版: 基本的な集計特徴量のみ使用
    """
    feature_start_dt = pd.to_datetime(feature_start)
    feature_end_dt = pd.to_datetime(feature_end)
    label_start_dt = pd.to_datetime(label_start)
    label_end_dt = pd.to_datetime(label_end)

    # 訓練期間のデータ
    feature_mask = (df['timestamp'] >= feature_start_dt) & (df['timestamp'] < feature_end_dt)
    feature_data = df[feature_mask].copy()

    label_mask = (df['timestamp'] >= label_start_dt) & (df['timestamp'] < label_end_dt)
    label_data = df[label_mask].copy()

    logger.info(f"  特徴量期間: {feature_start} to {feature_end} ({len(feature_data)} records)")
    logger.info(f"  ラベル期間: {label_start} to {label_end} ({len(label_data)} records)")

    # 訓練期間に活動した開発者
    developers = feature_data['email'].unique()
    logger.info(f"  訓練期間の開発者数: {len(developers)}")

    # IRLと同じく「評価期間に承諾があったか」でラベル付けする
    label_col = 'label' if 'label' in label_data.columns else None
    status_col = 'status' if 'status' in label_data.columns else None
    warned_no_accept_signal = False

    features_list = []

    for dev_email in developers:
        dev_train = feature_data[feature_data['email'] == dev_email]
        dev_eval = label_data[label_data['email'] == dev_email]

        if require_label_activity and len(dev_eval) == 0:
            continue

        if label_col:
            accepted_eval = dev_eval[dev_eval[label_col] == 1]
            label = 1 if len(accepted_eval) > 0 else 0
        elif status_col:
            accepted_eval = dev_eval[dev_eval[status_col].astype(str).str.upper() == 'MERGED']
            label = 1 if len(accepted_eval) > 0 else 0
        else:
            label = 1 if len(dev_eval) > 0 else 0
            if not warned_no_accept_signal:
                logger.warning("評価期間の承諾を示す列が見つからず、活動有無でラベル付与しています（IRLとは定義が異なる）")
                warned_no_accept_signal = True

        # 基本特徴量計算
        total_reviews = len(dev_train)

        # プロジェクト数（マルチプロジェクトのみ）
        if project_type == 'multi':
            project_count = dev_train['project'].nunique()
            projects = dev_train['project'].value_counts()
            main_project_ratio = projects.iloc[0] / total_reviews if len(projects) > 0 else 0
            project_distribution = 1.0 / project_count if project_count > 0 else 0
        else:
            project_count = 1
            main_project_ratio = 1.0
            project_distribution = 1.0

        # 時系列特徴量
        dates = dev_train['timestamp'].sort_values()
        experience_days = (feature_end_dt - dates.iloc[0]).days if len(dates) > 0 else 0

        if len(dates) > 1:
            gaps = dates.diff().dt.total_seconds() / 86400  # days
            avg_gap = gaps.mean()
            activity_freq = total_reviews / experience_days if experience_days > 0 else 0

            # トレンド（最近の活動増加/減少）
            half_point = len(dates) // 2
            recent_count = len(dates[half_point:])
            old_count = len(dates[:half_point])
            if recent_count > old_count * 1.2:
                trend = 1.0  # increasing
            elif recent_count < old_count * 0.8:
                trend = -1.0  # decreasing
            else:
                trend = 0.0  # stable
        else:
            avg_gap = 0
            activity_freq = 0
            trend = 0.0

        # コード品質・レビュー特徴量
        if 'status' in dev_train.columns:
            accepted = (dev_train['status'] == 'MERGED').sum()
            acceptance_rate = accepted / total_reviews if total_reviews > 0 else 0
        else:
            acceptance_rate = 0.5

        # コラボレーション
        if 'reviewer' in dev_train.columns:
            unique_reviewers = dev_train['reviewer'].nunique()
            collab_score = min(unique_reviewers / 10.0, 1.0)
        else:
            collab_score = 0.5

        # レビューサイズ（ファイル数など）
        if 'files_changed' in dev_train.columns:
            avg_files = dev_train['files_changed'].mean()
            review_size = min(avg_files / 10.0, 1.0)
        else:
            review_size = 0.5

        features = {
            'email': dev_email,
            # 状態特徴量
            'experience_days': experience_days,
            'total_changes': total_reviews,
            'total_reviews': total_reviews,
            'recent_activity_frequency': activity_freq,
            'avg_activity_gap': avg_gap,
            'activity_trend': trend,
            'collaboration_score': collab_score,
            'code_quality_score': acceptance_rate,
            'recent_acceptance_rate': acceptance_rate,
            'review_load': total_reviews / 90 if total_reviews > 0 else 0,  # per 90 days
            'project_count': project_count,
            'project_activity_distribution': project_distribution,
            'main_project_contribution_ratio': main_project_ratio,
            'cross_project_collaboration_score': (project_count - 1) / 10 if project_count > 1 else 0,
            # 行動特徴量
            'avg_action_intensity': activity_freq,
            'avg_collaboration': collab_score,
            'avg_response_time': avg_gap,
            'avg_review_size': review_size,
            'cross_project_action_ratio': (project_count - 1) / project_count if project_count > 1 else 0,
            # ラベル
            'true_label': label
        }

        features_list.append(features)

    features_df = pd.DataFrame(features_list)
    logger.info(f"  特徴量抽出完了: {len(features_df)} developers")
    logger.info(f"  継続率: {features_df['true_label'].mean()*100:.1f}%")

    return features_df


def build_cumulative_snapshots(
    df,
    train_start,
    train_end,
    label_months: int = 3,
    step_months: int = 1,
    project_type: str = 'multi'
):
    """累積特徴量 + 将来0-3mラベルを訓練期間内で繰り返し生成

    - 特徴量期間: train_start ～ cutoff（累積）
    - ラベル期間: cutoff ～ cutoff+label_months
    - label_end が train_end を超えるサンプルは除外
    - step_months間隔でcutoffを進めて複数スナップショットを作成
    """
    snapshots = []
    cutoff = pd.to_datetime(train_start)
    train_end_dt = pd.to_datetime(train_end)
    label_span = pd.DateOffset(months=label_months)
    step = pd.DateOffset(months=step_months)

    # まず1ステップ進めてラベル期間を確保（特徴量に少なくとも少しの履歴を持たせる）
    cutoff += step

    while True:
        label_start = cutoff
        label_end = cutoff + label_span
        if label_end > train_end_dt:
            break

        snap_df = extract_features_for_window(
            df,
            feature_start=train_start,
            feature_end=cutoff,
            label_start=label_start,
            label_end=label_end,
            project_type=project_type,
        )
        snap_df['cutoff'] = cutoff
        snapshots.append(snap_df)

        cutoff += step

    if snapshots:
        return pd.concat(snapshots, ignore_index=True)
    return pd.DataFrame()


def prepare_rf_features(df, project_type='multi'):
    """RFの入力特徴量を準備"""
    if project_type == 'multi':
        # マルチプロジェクト: 19次元
        features = [
            'experience_days', 'total_changes', 'total_reviews',
            'recent_activity_frequency', 'avg_activity_gap', 'activity_trend',
            'collaboration_score', 'code_quality_score', 'recent_acceptance_rate',
            'review_load', 'project_count', 'project_activity_distribution',
            'main_project_contribution_ratio', 'cross_project_collaboration_score',
            'avg_action_intensity', 'avg_collaboration', 'avg_response_time',
            'avg_review_size', 'cross_project_action_ratio'
        ]
    else:
        # Nova単体: 14次元（マルチプロジェクト特徴量を除外）
        features = [
            'experience_days', 'total_changes', 'total_reviews',
            'recent_activity_frequency', 'avg_activity_gap', 'activity_trend',
            'collaboration_score', 'code_quality_score', 'recent_acceptance_rate',
            'review_load',
            'avg_action_intensity', 'avg_collaboration', 'avg_response_time',
            'avg_review_size'
        ]

    X = df[features].fillna(0)
    y = df['true_label']

    return X, y


def train_and_evaluate_rf(X_train, y_train, X_eval, y_eval):
    """Random Forestを訓練・評価"""
    # クラス数チェック
    n_classes = len(np.unique(y_train))
    if n_classes < 2:
        logger.warning(f"  訓練データにクラスが{n_classes}つしかありません。スキップ")
        return None

    n_classes_eval = len(np.unique(y_eval))
    if n_classes_eval < 2:
        logger.warning(f"  評価データにクラスが{n_classes_eval}つしかありません。スキップ")
        return None

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
    auc_pr = average_precision_score(y_eval, y_pred_proba)
    precision = precision_score(y_eval, y_pred, zero_division=0)
    recall = recall_score(y_eval, y_pred, zero_division=0)
    accuracy = accuracy_score(y_eval, y_pred)

    # 混同行列
    tn, fp, fn, tp = confusion_matrix(y_eval, y_pred).ravel()

    return {
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


def evaluate_10_patterns(
    project_type='multi',
    rolling_train: bool = False,
    train_step_months: int = 1
):
    """10パターンでRFを評価"""
    logger.info("=" * 80)
    logger.info(f"Random Forest 10パターン評価 ({project_type.upper()})")
    logger.info("=" * 80)

    # データ読み込み
    df = load_data(project_type)

    # 10パターン定義
    patterns = [
        ("0-3m → 3-6m", "2020-07-01", "2020-10-01", "2020-10-01", "2021-01-01"),
        ("0-3m → 6-9m", "2020-07-01", "2020-10-01", "2021-01-01", "2021-04-01"),
        ("0-3m → 9-12m", "2020-07-01", "2020-10-01", "2021-04-01", "2021-07-01"),
        ("3-6m → 6-9m", "2020-10-01", "2021-01-01", "2021-01-01", "2021-04-01"),
        ("3-6m → 9-12m", "2020-10-01", "2021-01-01", "2021-04-01", "2021-07-01"),
        ("6-9m → 9-12m", "2021-01-01", "2021-04-01", "2022-10-01", "2023-01-01")
    ]

    all_results = []

    for pattern_name, train_start, train_end, eval_start, eval_end in patterns:
        logger.info(f"\n{'='*80}")
        logger.info(f"パターン: {pattern_name}")
        logger.info(f"{'='*80}")

        # IRL同様: 特徴量は履歴、ラベルは将来期間
        if rolling_train:
            train_features_df = build_cumulative_snapshots(
                df,
                train_start=train_start,
                train_end=train_end,
                label_months=3,
                step_months=train_step_months,
                project_type=project_type,
            )
        else:
            train_features_df = extract_features_for_window(
                df,
                feature_start=train_start,
                feature_end=train_end,
                label_start=eval_start,
                label_end=eval_end,
                project_type=project_type,
            )

        # 評価データ用: 履歴=eval期間, ラベル=その直後の同長さ期間
        eval_future_span = pd.to_datetime(eval_end) - pd.to_datetime(eval_start)
        eval_label_start = pd.to_datetime(eval_end)
        eval_label_end = eval_label_start + eval_future_span

        eval_features_df = extract_features_for_window(
            df,
            feature_start=eval_start,
            feature_end=eval_end,
            label_start=eval_label_start,
            label_end=eval_label_end,
            project_type=project_type,
        )

        if len(train_features_df) < 10 or len(eval_features_df) < 10:
            logger.warning(f"  サンプル数が少なすぎます。スキップ")
            continue

        # RF訓練・評価（訓練データと評価データを分離）
        X_train, y_train = prepare_rf_features(train_features_df, project_type)
        X_eval, y_eval = prepare_rf_features(eval_features_df, project_type)
        results = train_and_evaluate_rf(X_train, y_train, X_eval, y_eval)

        if results is None:
            logger.warning(f"  パターン {pattern_name} をスキップ")
            continue

        results['pattern'] = pattern_name
        results['train_window'] = pattern_name.split(' → ')[0]
        results['eval_window'] = pattern_name.split(' → ')[1]
        results['n_samples'] = len(X_eval)

        logger.info(f"\n結果:")
        logger.info(f"  F1:        {results['f1']:.4f}")
        logger.info(f"  AUC-ROC:   {results['auc_roc']:.4f}")
        logger.info(f"  AUC-PR:    {results['auc_pr']:.4f}")
        logger.info(f"  Recall:    {results['recall']:.4f}")
        logger.info(f"  Precision: {results['precision']:.4f}")
        logger.info(f"  TP={results['tp']}, TN={results['tn']}, FP={results['fp']}, FN={results['fn']}")

        all_results.append(results)

    return all_results


def save_results(results, project_type='multi'):
    """結果を保存"""
    output_dir = Path("/Users/kazuki-h/research/multiproject_research/outputs/analysis_data/rf_10patterns")
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV保存
    results_df = pd.DataFrame(results)
    csv_path = output_dir / f"rf_{project_type}_10patterns.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"\n✓ 結果をCSVに保存: {csv_path}")

    # JSON保存
    json_path = output_dir / f"rf_{project_type}_10patterns.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"✓ 結果をJSONに保存: {json_path}")

    # サマリー
    logger.info("\n" + "=" * 80)
    logger.info(f"{project_type.upper()} - サマリー ({len(results)}パターン)")
    logger.info("=" * 80)
    logger.info(f"平均 F1:        {results_df['f1'].mean():.4f}")
    logger.info(f"平均 AUC-ROC:   {results_df['auc_roc'].mean():.4f}")
    logger.info(f"平均 AUC-PR:    {results_df['auc_pr'].mean():.4f}")
    logger.info(f"平均 Recall:    {results_df['recall'].mean():.4f}")
    logger.info(f"平均 Precision: {results_df['precision'].mean():.4f}")
    logger.info(f"\n最高F1: {results_df['f1'].max():.4f} ({results_df.loc[results_df['f1'].idxmax(), 'pattern']})")
    logger.info(f"最低F1: {results_df['f1'].min():.4f} ({results_df.loc[results_df['f1'].idxmin(), 'pattern']})")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='RF 10パターン評価')
    parser.add_argument('--type', choices=['nova', 'multi', 'both'], default='both',
                       help='評価タイプ: nova, multi, or both')
    parser.add_argument('--rolling-train', action='store_true',
                       help='訓練期間内で累積特徴量+将来0-3mラベルを繰り返し生成')
    parser.add_argument('--train-step-months', type=int, default=1,
                       help='rolling-train時のcutoff間隔（月）')

    args = parser.parse_args()

    if args.type in ['nova', 'both']:
        logger.info("\n" + "=" * 80)
        logger.info("Nova単体プロジェクト評価開始")
        logger.info("=" * 80)
        nova_results = evaluate_10_patterns('nova', args.rolling_train, args.train_step_months)
        save_results(nova_results, 'nova')

    if args.type in ['multi', 'both']:
        logger.info("\n" + "=" * 80)
        logger.info("マルチプロジェクト評価開始")
        logger.info("=" * 80)
        multi_results = evaluate_10_patterns('multi', args.rolling_train, args.train_step_months)
        save_results(multi_results, 'multi')

    logger.info("\n" + "=" * 80)
    logger.info("全評価完了")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

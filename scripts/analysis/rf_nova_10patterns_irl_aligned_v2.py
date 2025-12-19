#!/usr/bin/env python3
"""
Random Forest 10パターン評価 - IRL完全一致版（Nova単体）

IRLの extract_evaluation_trajectories ロジックを完全に再現:
- 拡張期間（12ヶ月）チェック
- 除外ロジック（拡張期間にも依頼なし）
- label==1（承諾）でラベル付け
- 時間窓定義の完全一致

データソース: openstack_50proj_2021_2024.csv (openstack/nova フィルタ)
"""

import json
import logging
import sys
import time
from datetime import datetime
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


def load_data(project_filter='openstack/nova'):
    """Nova単体データを読み込み（IRLと同じデータソース）"""
    data_path = Path("/Users/kazuki-h/research/multiproject_research/data/review_requests_openstack_multi_5y_detail.csv")

    if not data_path.exists():
        raise FileNotFoundError(f"データファイルが見つかりません: {data_path}")

    logger.info(f"データ読み込み: {data_path}")
    df = pd.read_csv(data_path)

    # タイムスタンプ列を統一（IRLと同じく request_time を使用）
    if 'request_time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['request_time'])
    elif 'created' in df.columns:
        df['timestamp'] = pd.to_datetime(df['created'])
    elif 'context_date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['context_date'])
    else:
        raise ValueError("タイムスタンプ列が見つかりません")

    # email列を統一（IRLと同じく reviewer_email を使用）
    if 'reviewer_email' in df.columns:
        df['email'] = df['reviewer_email']
    elif 'email' not in df.columns:
        if 'developer_email' in df.columns:
            df['email'] = df['developer_email']
        else:
            raise ValueError("email列が見つかりません")

    logger.info(f"総レコード数: {len(df)}")

    # Nova単体にフィルタ
    if project_filter and 'project' in df.columns:
        df = df[df['project'] == project_filter].copy()
        logger.info(f"Nova単体にフィルタ: {len(df)} records")

    # label列の確認
    if 'label' not in df.columns:
        raise ValueError("label列が見つかりません。IRLと同じデータソースを使用してください。")

    logger.info(f"  承諾数 (label==1): {(df['label']==1).sum()} ({(df['label']==1).sum()/len(df)*100:.1f}%)")
    logger.info(f"  拒否数 (label==0): {(df['label']==0).sum()} ({(df['label']==0).sum()/len(df)*100:.1f}%)")

    return df


def extract_features_irl_aligned(
    df,
    feature_start,
    feature_end,
    label_start,
    label_end,
    extended_label_months=12,
    min_history_requests=3  # 最小履歴レビュー依頼数（IRLと一致）
):
    """
    IRL の extract_evaluation_trajectories ロジックを完全に再現

    Args:
        df: データフレーム
        feature_start: 特徴量計算開始日
        feature_end: 特徴量計算終了日
        label_start: ラベル計算開始日
        label_end: ラベル計算終了日
        extended_label_months: 拡張ラベル期間（月数）
        min_history_requests: 最小履歴レビュー依頼数（デフォルト3）

    Returns:
        特徴量DataFrame
    """
    feature_start_dt = pd.to_datetime(feature_start)
    feature_end_dt = pd.to_datetime(feature_end)
    label_start_dt = pd.to_datetime(label_start)
    label_end_dt = pd.to_datetime(label_end)

    logger.info(f"  特徴量期間: {feature_start} ～ {feature_end}")
    logger.info(f"  ラベル期間: {label_start} ～ {label_end}")
    logger.info(f"  拡張期間: {label_start} ～ {label_start_dt + pd.DateOffset(months=extended_label_months)}")

    # 特徴量期間のデータ
    feature_mask = (df['timestamp'] >= feature_start_dt) & (df['timestamp'] < feature_end_dt)
    feature_data = df[feature_mask].copy()

    # ラベル期間のデータ
    label_mask = (df['timestamp'] >= label_start_dt) & (df['timestamp'] < label_end_dt)
    label_data = df[label_mask].copy()

    # 拡張期間のデータ
    extended_label_end_dt = label_start_dt + pd.DateOffset(months=extended_label_months)
    extended_mask = (df['timestamp'] >= label_start_dt) & (df['timestamp'] < extended_label_end_dt)
    extended_data = df[extended_mask].copy()

    logger.info(f"  特徴量期間レコード数: {len(feature_data)}")
    logger.info(f"  ラベル期間レコード数: {len(label_data)}")
    logger.info(f"  拡張期間レコード数: {len(extended_data)}")

    # 特徴量期間に活動した開発者
    developers = feature_data['email'].unique()
    logger.info(f"  特徴量期間の開発者数: {len(developers)}")

    features_list = []
    excluded_no_extended_activity = 0  # 拡張期間にも依頼なし（除外）
    excluded_min_requests = 0  # 最小依頼数未満（除外）
    positive_count = 0
    negative_count = 0

    for dev_email in developers:
        dev_feature = feature_data[feature_data['email'] == dev_email]

        # 最小履歴レビュー依頼数チェック（IRLと同じ）
        if len(dev_feature) < min_history_requests:
            excluded_min_requests += 1
            continue

        dev_label = label_data[label_data['email'] == dev_email]

        # ラベル計算（IRLと同じロジック）
        if len(dev_label) == 0:
            # ラベル期間に依頼なし → 拡張期間をチェック
            dev_extended = extended_data[extended_data['email'] == dev_email]

            if len(dev_extended) == 0:
                # 拡張期間にも依頼なし → 除外（実質離脱者）
                excluded_no_extended_activity += 1
                continue

            # 拡張期間に依頼あり → 負例（依頼なし）
            label = 0
            negative_count += 1
        else:
            # ラベル期間に依頼あり → 承諾の有無で判定
            accepted = dev_label[dev_label['label'] == 1]
            label = 1 if len(accepted) > 0 else 0

            if label == 1:
                positive_count += 1
            else:
                negative_count += 1

        # 基本特徴量計算（簡易版）
        total_reviews = len(dev_feature)
        dates = dev_feature['timestamp'].sort_values()
        experience_days = (feature_end_dt - dates.iloc[0]).days if len(dates) > 0 else 0

        if len(dates) > 1:
            gaps = dates.diff().dt.total_seconds() / 86400
            avg_gap = gaps.mean()
            activity_freq = total_reviews / experience_days if experience_days > 0 else 0

            # トレンド
            half_point = len(dates) // 2
            recent_count = len(dates[half_point:])
            old_count = len(dates[:half_point])
            if recent_count > old_count * 1.2:
                trend = 1.0
            elif recent_count < old_count * 0.8:
                trend = -1.0
            else:
                trend = 0.0
        else:
            avg_gap = 0
            activity_freq = 0
            trend = 0.0

        # 承諾率
        if 'label' in dev_feature.columns:
            accepted_count = (dev_feature['label'] == 1).sum()
            acceptance_rate = accepted_count / total_reviews if total_reviews > 0 else 0
        else:
            acceptance_rate = 0.5

        # コラボレーション
        if 'owner_email' in dev_feature.columns:
            unique_collaborators = dev_feature['owner_email'].nunique()
            collab_score = min(unique_collaborators / 10.0, 1.0)
        else:
            collab_score = 0.5

        # レビューサイズ
        if 'change_files_count' in dev_feature.columns:
            avg_files = dev_feature['change_files_count'].mean()
            review_size = min(avg_files / 10.0, 1.0)
        else:
            review_size = 0.5

        # Nova単体特徴量（マルチプロジェクト特徴量を除外）
        features = {
            'email': dev_email,
            # 状態特徴量（10次元）
            'experience_days': experience_days,
            'total_changes': total_reviews,
            'total_reviews': total_reviews,
            'recent_activity_frequency': activity_freq,
            'avg_activity_gap': avg_gap,
            'activity_trend': trend,
            'collaboration_score': collab_score,
            'code_quality_score': acceptance_rate,
            'recent_acceptance_rate': acceptance_rate,
            'review_load': total_reviews / 90 if total_reviews > 0 else 0,
            # 行動特徴量（4次元）
            'avg_action_intensity': activity_freq,
            'avg_collaboration': collab_score,
            'avg_response_time': avg_gap,
            'avg_review_size': review_size,
            # ラベル
            'true_label': label
        }

        features_list.append(features)

    features_df = pd.DataFrame(features_list)

    logger.info(f"  特徴量抽出完了: {len(features_df)} developers")
    logger.info(f"  除外（最小依頼数 < {min_history_requests}）: {excluded_min_requests}")
    logger.info(f"  除外（拡張期間にも依頼なし）: {excluded_no_extended_activity}")
    logger.info(f"  正例（継続）: {positive_count} ({positive_count/len(features_df)*100:.1f}%)")
    logger.info(f"  負例（離脱）: {negative_count} ({negative_count/len(features_df)*100:.1f}%)")

    return features_df


def prepare_rf_features(df):
    """Nova単体用の特徴量を準備（14次元）"""
    features = [
        'experience_days', 'total_changes', 'total_reviews',
        'recent_activity_frequency', 'avg_activity_gap', 'activity_trend',
        'collaboration_score', 'code_quality_score', 'recent_acceptance_rate',
        'review_load',  # 状態10次元
        'avg_action_intensity', 'avg_collaboration', 'avg_response_time',
        'avg_review_size'  # 行動4次元
    ]

    X = df[features].fillna(0)
    y = df['true_label']

    return X, y


def train_and_evaluate_rf(X_train, y_train, X_eval, y_eval):
    """Random Forestを訓練・評価"""
    # クラス数チェック
    n_classes = len(np.unique(y_train))
    if n_classes < 2:
        logger.warning(f"  訓練データにクラスが{n_classes}つしかありません")
        return None

    n_classes_eval = len(np.unique(y_eval))
    if n_classes_eval < 2:
        logger.warning(f"  評価データにクラスが{n_classes_eval}つしかありません")
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


def evaluate_10_patterns(df, min_history_requests=3):
    """10パターンを評価（IRLと同じ）"""
    patterns = [
        {"train": ("2021-01-01", "2021-04-01"), "eval": ("2021-01-01", "2021-04-01"), "name": "0-3m → 0-3m"},
        {"train": ("2021-01-01", "2021-04-01"), "eval": ("2021-04-01", "2021-07-01"), "name": "0-3m → 3-6m"},
        {"train": ("2021-01-01", "2021-04-01"), "eval": ("2021-07-01", "2021-10-01"), "name": "0-3m → 6-9m"},
        {"train": ("2021-01-01", "2021-04-01"), "eval": ("2021-10-01", "2022-01-01"), "name": "0-3m → 9-12m"},
        {"train": ("2021-04-01", "2021-07-01"), "eval": ("2021-04-01", "2021-07-01"), "name": "3-6m → 3-6m"},
        {"train": ("2021-04-01", "2021-07-01"), "eval": ("2021-07-01", "2021-10-01"), "name": "3-6m → 6-9m"},
        {"train": ("2021-04-01", "2021-07-01"), "eval": ("2021-10-01", "2022-01-01"), "name": "3-6m → 9-12m"},
        {"train": ("2021-07-01", "2021-10-01"), "eval": ("2021-07-01", "2021-10-01"), "name": "6-9m → 6-9m"},
        {"train": ("2021-07-01", "2021-10-01"), "eval": ("2021-10-01", "2022-01-01"), "name": "6-9m → 9-12m"},
        {"train": ("2021-10-01", "2022-01-01"), "eval": ("2021-10-01", "2022-01-01"), "name": "9-12m → 9-12m"},
    ]

    results = []

    for i, pattern in enumerate(patterns, 1):
        logger.info("=" * 80)
        logger.info(f"パターン {i}/10: {pattern['name']}")
        logger.info("=" * 80)

        # 訓練データ特徴量抽出
        logger.info(f"訓練データ特徴量抽出...")
        train_features = extract_features_irl_aligned(
            df,
            feature_start=pattern['train'][0],
            feature_end=pattern['train'][1],
            label_start=pattern['train'][0],
            label_end=pattern['train'][1],
            min_history_requests=min_history_requests
        )

        # 評価データ特徴量抽出
        logger.info(f"評価データ特徴量抽出...")
        eval_features = extract_features_irl_aligned(
            df,
            feature_start=pattern['train'][0],
            feature_end=pattern['train'][1],
            label_start=pattern['eval'][0],
            label_end=pattern['eval'][1],
            min_history_requests=min_history_requests
        )

        # RF訓練・評価
        X_train, y_train = prepare_rf_features(train_features)
        X_eval, y_eval = prepare_rf_features(eval_features)

        logger.info(f"Random Forest訓練・評価...")
        result = train_and_evaluate_rf(X_train, y_train, X_eval, y_eval)

        if result is None:
            logger.warning(f"パターン {pattern['name']} をスキップ")
            continue

        result['pattern'] = pattern['name']
        result['train_window'] = f"{pattern['train'][0]} ～ {pattern['train'][1]}"
        result['eval_window'] = f"{pattern['eval'][0]} ～ {pattern['eval'][1]}"
        results.append(result)

        logger.info(f"  F1: {result['f1']:.4f}, AUC-ROC: {result['auc_roc']:.4f}, Recall: {result['recall']:.4f}")

    return results


def save_results(results):
    """結果を保存"""
    output_dir = Path("/Users/kazuki-h/research/multiproject_research/outputs/rf_nova_10patterns_irl_aligned_v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON保存
    json_path = output_dir / "all_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✓ 結果を保存: {json_path}")

    # サマリー
    results_df = pd.DataFrame(results)
    logger.info("\n" + "=" * 80)
    logger.info(f"Nova単体 RF (IRL完全一致版) - サマリー ({len(results)}パターン)")
    logger.info("=" * 80)
    logger.info(f"平均 F1:        {results_df['f1'].mean():.4f}")
    logger.info(f"平均 AUC-ROC:   {results_df['auc_roc'].mean():.4f}")
    logger.info(f"平均 Recall:    {results_df['recall'].mean():.4f}")
    logger.info(f"平均 Precision: {results_df['precision'].mean():.4f}")
    logger.info("=" * 80)

    # CSVマトリクス保存
    for metric in ['f1', 'auc_roc', 'precision', 'recall']:
        matrix = []
        train_periods = ['0-3m', '3-6m', '6-9m', '9-12m']
        eval_periods = ['0-3m', '3-6m', '6-9m', '9-12m']

        for train_p in train_periods:
            row = [train_p]
            for eval_p in eval_periods:
                pattern_name = f"{train_p} → {eval_p}"
                matching = [r for r in results if r['pattern'] == pattern_name]
                if matching:
                    row.append(matching[0][metric])
                else:
                    row.append(None)
            matrix.append(row)

        matrix_df = pd.DataFrame(matrix, columns=[''] + eval_periods)
        matrix_path = output_dir / f"matrix_{metric}.csv"
        matrix_df.to_csv(matrix_path, index=False)
        logger.info(f"✓ マトリクス保存: {matrix_path}")


def main():
    MIN_HISTORY_REQUESTS = 1  # IRLと一致（試行: 1に変更してサンプル数を確認）

    logger.info("=" * 80)
    logger.info("Random Forest 10パターン評価 - IRL完全一致版（Nova単体）")
    logger.info("特徴量: 状態10次元 + 行動4次元 = 14次元")
    logger.info("ラベル定義: IRLと完全一致（拡張期間チェック、除外ロジック）")
    logger.info(f"最小履歴依頼数: {MIN_HISTORY_REQUESTS}")
    logger.info("=" * 80)

    # データ読み込み
    df = load_data(project_filter='openstack/nova')

    # 10パターン評価
    results = evaluate_10_patterns(df, min_history_requests=MIN_HISTORY_REQUESTS)

    # 結果保存
    save_results(results)

    logger.info("\n完了！")


if __name__ == '__main__':
    main()

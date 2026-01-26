#!/usr/bin/env python3
"""
Random Forest 10パターン評価 - 統一特徴量版（Nova単体）

IRLとRFで完全に同じ特徴量を使用:
- 共通の特徴量抽出モジュール (src/review_predictor/features/common_features.py)
- 計算方法の統一
- 正規化方法の統一
- データソースの統一

データソース: review_requests_openstack_multi_5y_detail.csv (openstack/nova フィルタ)
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

# 共通特徴量モジュールをインポート
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.review_predictor.features.common_features import (
    extract_common_features,
    FEATURE_NAMES,
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


def extract_features_unified(
    df,
    cutoff_date,
    history_window_months,
    future_window_start_months,
    future_window_end_months,
    extended_label_months=12,
    min_history_requests=0  # 最小履歴レビュー依頼数（IRLと一致: デフォルト0）
):
    """
    統一された特徴量を抽出（IRLと完全一致）

    Args:
        df: データフレーム
        cutoff_date: カットオフ日（訓練終了日/評価開始基準日）
        history_window_months: 履歴期間（月数、デフォルト12）
        future_window_start_months: 将来窓の開始（月数）
        future_window_end_months: 将来窓の終了（月数）
        extended_label_months: 拡張ラベル期間（月数）
        min_history_requests: 最小履歴レビュー依頼数（デフォルト0）

    Returns:
        特徴量DataFrame
    """
    cutoff_dt = pd.to_datetime(cutoff_date)

    # IRLと同じロジック: cutoff_dateを基準に期間を設定
    feature_start_dt = cutoff_dt - pd.DateOffset(months=history_window_months)
    feature_end_dt = cutoff_dt
    label_start_dt = cutoff_dt + pd.DateOffset(months=future_window_start_months)
    label_end_dt = cutoff_dt + pd.DateOffset(months=future_window_end_months)

    logger.info(f"  カットオフ日: {cutoff_date}")
    logger.info(f"  履歴期間: {feature_start_dt} ～ {feature_end_dt} ({history_window_months}ヶ月)")
    logger.info(f"  将来窓（ラベル期間）: {label_start_dt} ～ {label_end_dt} ({future_window_start_months}-{future_window_end_months}ヶ月)")
    logger.info(f"  拡張期間: {label_start_dt} ～ {cutoff_dt + pd.DateOffset(months=extended_label_months)}")

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

        # ========================================
        # 統一特徴量抽出（共通モジュールを使用）
        # ========================================
        features = extract_common_features(
            df=df,
            email=dev_email,
            feature_start=feature_start_dt,
            feature_end=feature_end_dt,
            normalize=False  # RFでは正規化しない（生の値を使用）
        )

        # emailとラベルを追加
        features['email'] = dev_email
        features['true_label'] = label

        features_list.append(features)

    features_df = pd.DataFrame(features_list)

    logger.info(f"  特徴量抽出完了: {len(features_df)} developers")
    logger.info(f"  除外（最小依頼数 < {min_history_requests}）: {excluded_min_requests}")
    logger.info(f"  除外（拡張期間にも依頼なし）: {excluded_no_extended_activity}")
    logger.info(f"  正例（継続）: {positive_count} ({positive_count/len(features_df)*100:.1f}%)")
    logger.info(f"  負例（離脱）: {negative_count} ({negative_count/len(features_df)*100:.1f}%)")

    return features_df


def prepare_rf_features(df):
    """統一された特徴量を準備（14次元）"""
    X = df[FEATURE_NAMES].fillna(0)
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


def evaluate_10_patterns(df, min_history_requests=0):
    """10パターンを評価（IRLと完全一致）"""

    # IRLと同じ設定
    CUTOFF_DATE = "2023-01-01"  # 訓練終了日/評価開始基準日
    HISTORY_WINDOW_MONTHS = 12  # 履歴期間（12ヶ月）

    # 訓練期間と評価期間のパターン定義（将来窓ベース）
    patterns = [
        # 訓練: 0-3m, 評価: 0-3m/3-6m/6-9m/9-12m
        {"train_period": "0-3m", "eval_period": "0-3m", "train_start": 0, "train_end": 3, "eval_start": 0, "eval_end": 3},
        {"train_period": "0-3m", "eval_period": "3-6m", "train_start": 0, "train_end": 3, "eval_start": 3, "eval_end": 6},
        {"train_period": "0-3m", "eval_period": "6-9m", "train_start": 0, "train_end": 3, "eval_start": 6, "eval_end": 9},
        {"train_period": "0-3m", "eval_period": "9-12m", "train_start": 0, "train_end": 3, "eval_start": 9, "eval_end": 12},
        # 訓練: 3-6m, 評価: 3-6m/6-9m/9-12m
        {"train_period": "3-6m", "eval_period": "3-6m", "train_start": 3, "train_end": 6, "eval_start": 3, "eval_end": 6},
        {"train_period": "3-6m", "eval_period": "6-9m", "train_start": 3, "train_end": 6, "eval_start": 6, "eval_end": 9},
        {"train_period": "3-6m", "eval_period": "9-12m", "train_start": 3, "train_end": 6, "eval_start": 9, "eval_end": 12},
        # 訓練: 6-9m, 評価: 6-9m/9-12m
        {"train_period": "6-9m", "eval_period": "6-9m", "train_start": 6, "train_end": 9, "eval_start": 6, "eval_end": 9},
        {"train_period": "6-9m", "eval_period": "9-12m", "train_start": 6, "train_end": 9, "eval_start": 9, "eval_end": 12},
        # 訓練: 9-12m, 評価: 9-12m
        {"train_period": "9-12m", "eval_period": "9-12m", "train_start": 9, "train_end": 12, "eval_start": 9, "eval_end": 12},
    ]

    results = []

    for i, pattern in enumerate(patterns, 1):
        pattern_name = f"{pattern['train_period']} → {pattern['eval_period']}"
        logger.info("=" * 80)
        logger.info(f"パターン {i}/10: {pattern_name}")
        logger.info("=" * 80)

        # 訓練データ特徴量抽出
        logger.info(f"訓練データ特徴量抽出...")
        train_features = extract_features_unified(
            df,
            cutoff_date=CUTOFF_DATE,
            history_window_months=HISTORY_WINDOW_MONTHS,
            future_window_start_months=pattern['train_start'],
            future_window_end_months=pattern['train_end'],
            min_history_requests=min_history_requests
        )

        # 評価データ特徴量抽出
        logger.info(f"評価データ特徴量抽出...")
        eval_features = extract_features_unified(
            df,
            cutoff_date=CUTOFF_DATE,
            history_window_months=HISTORY_WINDOW_MONTHS,
            future_window_start_months=pattern['eval_start'],
            future_window_end_months=pattern['eval_end'],
            min_history_requests=min_history_requests
        )

        # RF訓練・評価
        X_train, y_train = prepare_rf_features(train_features)
        X_eval, y_eval = prepare_rf_features(eval_features)

        logger.info(f"Random Forest訓練・評価...")
        result = train_and_evaluate_rf(X_train, y_train, X_eval, y_eval)

        if result is None:
            logger.warning(f"パターン {pattern_name} をスキップ")
            continue

        result['pattern'] = pattern_name
        result['train_period'] = pattern['train_period']
        result['eval_period'] = pattern['eval_period']
        result['train_window'] = f"{pattern['train_start']}-{pattern['train_end']}ヶ月"
        result['eval_window'] = f"{pattern['eval_start']}-{pattern['eval_end']}ヶ月"
        results.append(result)

        logger.info(f"  F1: {result['f1']:.4f}, AUC-ROC: {result['auc_roc']:.4f}, Recall: {result['recall']:.4f}")

    return results


def save_results(results):
    """結果を保存"""
    output_dir = Path("/Users/kazuki-h/research/multiproject_research/outputs/rf_nova_10patterns_unified_features")
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON保存
    json_path = output_dir / "all_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✓ 結果を保存: {json_path}")

    # サマリー
    results_df = pd.DataFrame(results)
    logger.info("\n" + "=" * 80)
    logger.info(f"Nova単体 RF (統一特徴量版) - サマリー ({len(results)}パターン)")
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
    MIN_HISTORY_REQUESTS = 0  # IRLと一致（デフォルト0）

    logger.info("=" * 80)
    logger.info("Random Forest 10パターン評価 - IRLと完全統一版（Nova単体）")
    logger.info("=" * 80)
    logger.info("【実験条件】")
    logger.info("  カットオフ日: 2023-01-01（訓練終了日/評価開始基準日）")
    logger.info("  履歴期間: 2022-01-01 ～ 2023-01-01（12ヶ月）")
    logger.info("  訓練期間: 将来窓 0-3m, 3-6m, 6-9m, 9-12m")
    logger.info("  評価期間: 将来窓 0-3m, 3-6m, 6-9m, 9-12m")
    logger.info("  特徴量: IRLと完全統一（共通モジュール使用）")
    logger.info("  ラベル定義: IRLと完全一致（拡張期間チェック、除外ロジック）")
    logger.info(f"  最小履歴依頼数: {MIN_HISTORY_REQUESTS}")
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

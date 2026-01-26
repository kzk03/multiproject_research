#!/usr/bin/env python3
"""
Random Forest クロス評価 - IRLと完全統一版（Nova単体、データリークなし）

IRLの実験設計と完全に一致:
- 4訓練期間 × 4評価期間 = 16パターン
- 同じカットオフ日（2023-01-01）
- 同じ特徴量（共通モジュール使用）
- 同じラベル定義（拡張期間チェック、除外ロジック）
- 同じディレクトリ構造（比較しやすい）

【データリーク防止】重要な修正！
- 訓練データ: 2021年の訓練期間から特徴量を抽出（例: 0-3m = 2021-01-01 ～ 2021-04-01）
- 評価データ: 2022-2023年の12ヶ月から特徴量を抽出（2022-01-01 ～ 2023-01-01）
- ラベル: 両方とも2023年の評価期間を使用（例: 0-3m = 2023-01-01 ～ 2023-04-01）
→ これにより訓練と評価で開発者の重複を防ぐ

【IRLとの比較ポイント】
1. IRLは時系列（LSTM）を考慮、RFは時系列を考慮しない
2. 両者の精度差が時系列の効果を示す
3. 期間変更による影響の違いを分析

データソース: review_requests_openstack_multi_5y_detail.csv (openstack/nova フィルタ)
"""

import argparse
import json
import logging
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

# 共通特徴量モジュールをインポート
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.review_predictor.features.common_features import (
    FEATURE_NAMES,
    extract_common_features,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========================================
# IRLと同じ実験設定（時間分離・履歴窓・最小依頼数を揃える）
# ========================================
TRAIN_BASE_START = "2021-01-01"  # 訓練特徴量のベース開始日
TRAIN_CUTOFF_DATE = "2022-01-01"  # 訓練ラベル用の基準日（学習用）
EVAL_CUTOFF_DATE = "2023-01-01"   # 評価ラベル用の基準日（評価用）
HISTORY_WINDOW_MONTHS = 24         # IRLに合わせて評価履歴窓を24ヶ月に統一
EXTENDED_LABEL_MONTHS = 12         # 拡張ラベル期間（12ヶ月）
MIN_HISTORY_REQUESTS = 3           # 最小履歴依頼数をIRLと合わせる

# 訓練期間と評価期間の定義（将来窓の月数オフセット）
# 訓練期間は2021年のベースから計算、評価期間は2023年のcutoffから計算
TRAIN_PERIODS = {
    '0-3m': (0, 3),
    '3-6m': (3, 6),
    '6-9m': (6, 9),
    '9-12m': (9, 12),
}

EVAL_PERIODS = {
    '0-3m': (0, 3),
    '3-6m': (3, 6),
    '6-9m': (6, 9),
    '9-12m': (9, 12),
}


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
    min_history_requests=0,
    is_training=False,
    train_start=None,
    train_end=None
):
    """
    統一された特徴量を抽出（IRLと完全一致、データリークなし）

    Args:
        df: データフレーム
        cutoff_date: カットオフ日（訓練終了日/評価開始基準日）
        history_window_months: 履歴期間（月数、デフォルト12、評価時のみ使用）
        future_window_start_months: 将来窓の開始（月数）
        future_window_end_months: 将来窓の終了（月数）
        extended_label_months: 拡張ラベル期間（月数）
        min_history_requests: 最小履歴レビュー依頼数（デフォルト0）
        is_training: 訓練データの場合True（特徴量抽出期間が異なる）
        train_start: 訓練期間開始日（is_training=Trueの場合必須）
        train_end: 訓練期間終了日（is_training=Trueの場合必須）

    Returns:
        特徴量DataFrame
    """
    cutoff_dt = pd.to_datetime(cutoff_date)

    # データリークを防ぐために、訓練と評価で異なる期間を使用
    if is_training:
        # 訓練時: 2021年の訓練期間を特徴量抽出に使用
        if train_start is None or train_end is None:
            raise ValueError("is_training=Trueの場合、train_startとtrain_endが必須です")
        feature_start_dt = train_start
        feature_end_dt = train_end
    else:
        # 評価時: cutoff_date（2023-01-01）の12ヶ月前から特徴量を抽出
        feature_start_dt = cutoff_dt - pd.DateOffset(months=history_window_months)
        feature_end_dt = cutoff_dt

    # ラベルは両方とも2023年の評価期間を使用
    label_start_dt = cutoff_dt + pd.DateOffset(months=future_window_start_months)
    label_end_dt = cutoff_dt + pd.DateOffset(months=future_window_end_months)

    if is_training:
        logger.info(f"  【訓練データ】特徴量期間: {feature_start_dt} ～ {feature_end_dt}")
    else:
        logger.info(f"  【評価データ】特徴量期間: {feature_start_dt} ～ {feature_end_dt} ({history_window_months}ヶ月)")
    logger.info(f"  ラベル期間: {label_start_dt} ～ {label_end_dt} ({future_window_start_months}-{future_window_end_months}ヶ月)")

    # 特徴量期間のデータ
    feature_mask = (df['timestamp'] >= feature_start_dt) & (df['timestamp'] < feature_end_dt)
    feature_data = df[feature_mask].copy()

    # ラベル期間のデータ
    label_mask = (df['timestamp'] >= label_start_dt) & (df['timestamp'] < label_end_dt)
    label_data = df[label_mask].copy()

    # 拡張期間のデータ
    extended_label_end_dt = cutoff_dt + pd.DateOffset(months=extended_label_months)
    extended_mask = (df['timestamp'] >= label_start_dt) & (df['timestamp'] < extended_label_end_dt)
    extended_data = df[extended_mask].copy()

    # 特徴量期間に活動した開発者
    developers = feature_data['email'].unique()
    logger.info(f"  履歴期間の開発者数: {len(developers)}")

    features_list = []
    excluded_no_extended_activity = 0
    excluded_min_requests = 0
    positive_count = 0
    negative_count = 0

    for dev_email in developers:
        dev_feature = feature_data[feature_data['email'] == dev_email]

        # 最小履歴レビュー依頼数チェック
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

        # 統一特徴量抽出
        features = extract_common_features(
            df=df,
            email=dev_email,
            feature_start=feature_start_dt,
            feature_end=feature_end_dt,
            normalize=False  # RFでは正規化しない
        )

        # IRLのpredictions.csvと同じ列を追加
        features['reviewer_email'] = dev_email
        features['true_label'] = label
        features['history_request_count'] = len(dev_feature)
        features['history_acceptance_rate'] = (dev_feature['label'] == 1).sum() / len(dev_feature) if len(dev_feature) > 0 else 0.0
        features['eval_request_count'] = len(dev_label)
        features['eval_accepted_count'] = (dev_label['label'] == 1).sum() if len(dev_label) > 0 else 0
        features['eval_rejected_count'] = (dev_label['label'] == 0).sum() if len(dev_label) > 0 else 0

        features_list.append(features)

    features_df = pd.DataFrame(features_list)

    logger.info(f"  特徴量抽出完了: {len(features_df)} developers")
    logger.info(f"  除外（最小依頼数 < {min_history_requests}）: {excluded_min_requests}")
    logger.info(f"  除外（拡張期間にも依頼なし）: {excluded_no_extended_activity}")
    logger.info(f"  正例（継続）: {positive_count} ({positive_count/max(len(features_df),1)*100:.1f}%)")
    logger.info(f"  負例（離脱）: {negative_count} ({negative_count/max(len(features_df),1)*100:.1f}%)")

    return features_df


def train_rf_model(X_train, y_train, random_state=777):
    """Random Forestモデルを訓練"""
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1
    )

    start_time = time.time()
    rf.fit(X_train, y_train)
    train_time = time.time() - start_time

    return rf, train_time


def find_optimal_threshold(y_true, y_pred_proba):
    """評価データを使わず訓練データで閾値を決定（IRLと同様にF1最大化）。"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-12)
    best_idx = int(np.argmax(f1_scores))
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    return {
        'threshold': float(best_threshold),
        'precision': float(precision[best_idx]),
        'recall': float(recall[best_idx]),
        'f1': float(f1_scores[best_idx])
    }


def plot_feature_importances(importances_df, output_path, top_n=30):
    """特徴量重要度を折れ線グラフで保存（Giniベース）。"""
    top_df = importances_df.head(top_n)
    plt.figure(figsize=(10, 4))
    plt.plot(top_df['feature'], top_df['importance'], marker='o')
    plt.xticks(rotation=75, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def evaluate_rf_model(rf, X_eval, y_eval, threshold=0.5):
    """Random Forestモデルを評価（指定閾値で二値化）。"""
    # クラス数チェック
    n_classes = len(np.unique(y_eval))
    if n_classes < 2:
        logger.warning(f"  評価データにクラスが{n_classes}つしかありません")
        return None

    start_time = time.time()
    y_pred_proba = rf.predict_proba(X_eval)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
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
        'auc_roc': float(auc_roc),
        'auc_pr': float(auc_pr),
        'f1_score': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'accuracy': float(accuracy),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'predict_time': float(predict_time),
        'eval_samples': len(X_eval),
        'positive_count': int((y_eval == 1).sum()),
        'negative_count': int((y_eval == 0).sum()),
        'total_count': len(y_eval),
        'threshold': float(threshold),
    }


def save_predictions(features_df, y_pred_proba, y_pred, output_path):
    """予測結果を保存（IRLと同じフォーマット）"""
    predictions_df = features_df.copy()
    predictions_df['predicted_prob'] = y_pred_proba
    predictions_df['predicted_binary'] = y_pred

    # IRLと同じ列順序
    columns_order = [
        'reviewer_email',
        'predicted_prob',
        'true_label',
        'history_request_count',
        'history_acceptance_rate',
        'eval_request_count',
        'eval_accepted_count',
        'eval_rejected_count',
        'predicted_binary',
    ]

    # 特徴量も含める
    for col in FEATURE_NAMES:
        if col in predictions_df.columns:
            columns_order.append(col)

    predictions_df[columns_order].to_csv(output_path, index=False)
    logger.info(f"  ✓ 予測結果保存: {output_path}")


def cross_evaluate_rf(df, output_dir, random_state=777):
    """
    クロス評価を実行（IRLと同じ16パターン）

    4訓練期間 × 4評価期間 = 16通り
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # 各訓練期間でモデルを訓練
    for train_period, (train_start_offset, train_end_offset) in TRAIN_PERIODS.items():
        logger.info("=" * 80)
        logger.info(f"訓練期間: {train_period} ({train_start_offset}-{train_end_offset}ヶ月)")
        logger.info("=" * 80)

        train_dir = output_dir / f"train_{train_period}"
        train_dir.mkdir(parents=True, exist_ok=True)

        # 訓練期間の実際の日付を計算（2021年ベース）
        train_base_dt = pd.to_datetime(TRAIN_BASE_START)
        train_start_dt = train_base_dt + pd.DateOffset(months=train_start_offset)
        train_end_dt = train_base_dt + pd.DateOffset(months=train_end_offset)

        logger.info(f"訓練期間の実際の日付: {train_start_dt.date()} ～ {train_end_dt.date()}")

        # 訓練データ抽出（2021年の訓練期間から特徴量を抽出）
        logger.info("訓練データ特徴量抽出...")
        train_features = extract_features_unified(
            df,
            cutoff_date=TRAIN_CUTOFF_DATE,
            history_window_months=HISTORY_WINDOW_MONTHS,
            future_window_start_months=train_start_offset,  # ラベル期間のオフセット
            future_window_end_months=train_end_offset,  # ラベル期間のオフセット
            extended_label_months=EXTENDED_LABEL_MONTHS,
            min_history_requests=MIN_HISTORY_REQUESTS,
            is_training=True,  # 訓練データフラグ
            train_start=train_start_dt,  # 2021年の訓練開始日
            train_end=train_end_dt  # 2021年の訓練終了日
        )

        X_train = train_features[FEATURE_NAMES].fillna(0)
        y_train = train_features['true_label']

        # クラス数チェック
        n_classes = len(np.unique(y_train))
        if n_classes < 2:
            logger.warning(f"訓練データにクラスが{n_classes}つしかありません - スキップ")
            continue

        # モデル訓練
        logger.info("Random Forest訓練...")
        rf_model, train_time = train_rf_model(X_train, y_train, random_state=random_state)

        # 特徴量重要度を保存（Gini係数ベース）
        importances = pd.DataFrame({
            'feature': FEATURE_NAMES,
            'importance': rf_model.feature_importances_,
        }).sort_values('importance', ascending=False)
        importances.to_csv(train_dir / "feature_importances.csv", index=False)
        plot_feature_importances(importances, train_dir / "feature_importances.png")

        # 訓練データでの予測・閾値決定（F1最大化、IRLと同様に学習データのみで決定）
        train_pred_proba = rf_model.predict_proba(X_train)[:, 1]
        train_threshold_info = find_optimal_threshold(y_train, train_pred_proba)
        train_threshold = train_threshold_info['threshold']
        train_pred = (train_pred_proba >= train_threshold).astype(int)

        # 訓練結果を保存
        save_predictions(
            train_features,
            train_pred_proba,
            train_pred,
            train_dir / "predictions.csv"
        )

        # 対角線評価（train_period == eval_period）
        diagonal_metrics = evaluate_rf_model(rf_model, X_train, y_train, threshold=train_threshold)
        if diagonal_metrics:
            diagonal_metrics['train_time'] = train_time
            diagonal_metrics['train_samples'] = len(X_train)
            diagonal_metrics['threshold_source'] = 'train_f1_max'
            diagonal_metrics['train_optimal_threshold'] = train_threshold
            diagonal_metrics['train_optimal_f1'] = train_threshold_info['f1']
            diagonal_metrics['train_optimal_precision'] = train_threshold_info['precision']
            diagonal_metrics['train_optimal_recall'] = train_threshold_info['recall']

            with open(train_dir / "metrics.json", 'w') as f:
                json.dump(diagonal_metrics, f, indent=2)
            logger.info(f"  ✓ 訓練メトリクス保存: AUC-ROC={diagonal_metrics['auc_roc']:.4f}")

        # 各評価期間で評価
        for eval_period, (eval_start_offset, eval_end_offset) in EVAL_PERIODS.items():
            logger.info("-" * 80)
            logger.info(f"評価期間: {eval_period} ({eval_start_offset}-{eval_end_offset}ヶ月)")

            eval_dir = train_dir / f"eval_{eval_period}"
            eval_dir.mkdir(parents=True, exist_ok=True)

            # 評価データ抽出（2022-2023年の12ヶ月から特徴量を抽出）
            logger.info("評価データ特徴量抽出...")
            eval_features = extract_features_unified(
                df,
                cutoff_date=EVAL_CUTOFF_DATE,
                history_window_months=HISTORY_WINDOW_MONTHS,
                future_window_start_months=eval_start_offset,  # ラベル期間のオフセット
                future_window_end_months=eval_end_offset,  # ラベル期間のオフセット
                extended_label_months=EXTENDED_LABEL_MONTHS,
                min_history_requests=MIN_HISTORY_REQUESTS,
                is_training=False  # 評価データフラグ（12ヶ月の履歴窓を使用）
            )

            X_eval = eval_features[FEATURE_NAMES].fillna(0)
            y_eval = eval_features['true_label']

            # 評価
            logger.info("Random Forest評価...")
            eval_metrics = evaluate_rf_model(rf_model, X_eval, y_eval, threshold=train_threshold)

            if eval_metrics is None:
                logger.warning(f"  評価スキップ: {train_period} → {eval_period}")
                continue

            # 予測結果を保存
            eval_pred_proba = rf_model.predict_proba(X_eval)[:, 1]
            eval_pred = (eval_pred_proba >= train_threshold).astype(int)
            save_predictions(
                eval_features,
                eval_pred_proba,
                eval_pred,
                eval_dir / "predictions.csv"
            )

            # メトリクスを保存
            with open(eval_dir / "metrics.json", 'w') as f:
                json.dump(eval_metrics, f, indent=2)

            logger.info(f"  ✓ {train_period} → {eval_period}: AUC-ROC={eval_metrics['auc_roc']:.4f}, F1={eval_metrics['f1_score']:.4f}")

            # 結果を収集
            pattern_key = f"{train_period} → {eval_period}"
            all_results[pattern_key] = eval_metrics

    return all_results


def create_summary_matrices(all_results, output_dir):
    """サマリーマトリクスを作成（IRLと同じフォーマット）"""
    output_dir = Path(output_dir)

    metrics = ['auc_roc', 'auc_pr', 'f1_score', 'precision', 'recall']
    metric_names = {
        'auc_roc': 'AUC_ROC',
        'auc_pr': 'AUC_PR',
        'f1_score': 'F1',
        'precision': 'PRECISION',
        'recall': 'RECALL',
    }

    train_periods = ['0-3m', '3-6m', '6-9m', '9-12m']
    eval_periods = ['0-3m', '3-6m', '6-9m', '9-12m']

    for metric in metrics:
        # マトリクス作成
        matrix = []
        for train_p in train_periods:
            row = [train_p]
            for eval_p in eval_periods:
                pattern_key = f"{train_p} → {eval_p}"
                if pattern_key in all_results:
                    row.append(all_results[pattern_key][metric])
                else:
                    row.append(None)
            matrix.append(row)

        # CSV保存
        matrix_df = pd.DataFrame(matrix, columns=[''] + eval_periods)
        matrix_path = output_dir / f"matrix_{metric_names[metric]}.csv"
        matrix_df.to_csv(matrix_path, index=False)
        logger.info(f"✓ マトリクス保存: {matrix_path}")


def main():
    parser = argparse.ArgumentParser(description="Random Forestクロス評価（IRL統一版）")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/rf_nova_cross_eval_unified"),
        help="結果を保存するディレクトリ"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=777,
        help="RandomForestのrandom_state"
    )
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Random Forest クロス評価 - IRLと完全統一版（Nova単体）")
    logger.info("=" * 80)
    logger.info("【実験条件】")
    logger.info(f"  学習用カットオフ日: {TRAIN_CUTOFF_DATE}（学習ラベル基準日）")
    logger.info(f"  評価用カットオフ日: {EVAL_CUTOFF_DATE}（評価ラベル基準日）")
    logger.info(f"  評価履歴期間: {HISTORY_WINDOW_MONTHS}ヶ月（IRLに合わせて24ヶ月）")
    logger.info("  訓練期間: 将来窓 0-3m, 3-6m, 6-9m, 9-12m")
    logger.info("  評価期間: 将来窓 0-3m, 3-6m, 6-9m, 9-12m")
    logger.info("  クロス評価: 4 × 4 = 16パターン")
    logger.info("  特徴量: IRLと完全統一（共通モジュール使用）")
    logger.info("  ラベル定義: IRLと完全一致（拡張期間チェック、除外ロジック）")
    logger.info(f"  最小履歴依頼数: {MIN_HISTORY_REQUESTS}（IRL合わせ）")
    logger.info(f"  random_state: {args.random_state}")
    logger.info("")
    logger.info("【IRLとの比較】")
    logger.info("  IRL: 時系列（LSTM）を考慮")
    logger.info("  RF:  時系列を考慮しない（スナップショット特徴量のみ）")
    logger.info("=" * 80)

    # データ読み込み
    df = load_data(project_filter='openstack/nova')

    # クロス評価実行
    output_dir = args.output_dir
    all_results = cross_evaluate_rf(df, output_dir, random_state=args.random_state)

    # サマリーマトリクス作成
    logger.info("\n" + "=" * 80)
    logger.info("サマリーマトリクス作成")
    logger.info("=" * 80)
    create_summary_matrices(all_results, output_dir)

    # 全体サマリー
    logger.info("\n" + "=" * 80)
    logger.info(f"完了！結果は {output_dir} に保存されました")
    logger.info("=" * 80)

    if all_results:
        all_auc_roc = [r['auc_roc'] for r in all_results.values()]
        all_f1 = [r['f1_score'] for r in all_results.values()]
        logger.info(f"平均 AUC-ROC: {np.mean(all_auc_roc):.4f}")
        logger.info(f"平均 F1:      {np.mean(all_f1):.4f}")
        logger.info(f"評価パターン数: {len(all_results)}")


if __name__ == '__main__':
    main()

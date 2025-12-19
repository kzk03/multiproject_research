"""
Random Forest Nova単体評価 - 案1: スライディングウィンドウ (10パターン版)

訓練データ: スライディングウィンドウで生成 (~1000サンプル)
評価データ: 2023年の各期間
10パターン: 各訓練期間 → 各評価期間（過去→未来 or 同期間のみ）
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_samples_at_cutoff(
    df,
    cutoff_date,
    history_months=12,
    future_months=3,
    extended_months=12,
    min_history_requests=1
):
    """1つのcutoff時点でサンプルを抽出"""
    cutoff_dt = pd.to_datetime(cutoff_date)

    # 履歴期間
    history_start = cutoff_dt - pd.DateOffset(months=history_months)
    history_end = cutoff_dt

    # ラベル期間
    label_start = cutoff_dt
    label_end = cutoff_dt + pd.DateOffset(months=future_months)

    # 拡張期間
    extended_end = cutoff_dt + pd.DateOffset(months=extended_months)

    # データ抽出
    history_mask = (df['timestamp'] >= history_start) & (df['timestamp'] < history_end)
    history_data = df[history_mask].copy()

    label_mask = (df['timestamp'] >= label_start) & (df['timestamp'] < label_end)
    label_data = df[label_mask].copy()

    extended_mask = (df['timestamp'] >= label_start) & (df['timestamp'] < extended_end)
    extended_data = df[extended_mask].copy()

    # 開発者抽出
    developers = history_data['reviewer_email'].unique()

    samples = []

    for dev_email in developers:
        dev_history = history_data[history_data['reviewer_email'] == dev_email]

        # 最小依頼数フィルタ
        if len(dev_history) < min_history_requests:
            continue

        dev_label = label_data[label_data['reviewer_email'] == dev_email]

        # ラベル計算（IRLと同じ）
        if len(dev_label) == 0:
            dev_extended = extended_data[extended_data['reviewer_email'] == dev_email]
            if len(dev_extended) == 0:
                continue  # 除外
            label = 0
        else:
            accepted = dev_label[dev_label['label'] == 1]
            label = 1 if len(accepted) > 0 else 0

        # 特徴量計算
        total_reviews = len(dev_history)
        dates = dev_history['timestamp'].sort_values()
        experience_days = (history_end - dates.iloc[0]).days if len(dates) > 0 else 0

        if experience_days > 0:
            activity_freq = total_reviews / (experience_days / 30.0)
        else:
            activity_freq = 0

        accepted_count = (dev_history['label'] == 1).sum()
        acceptance_rate = accepted_count / total_reviews if total_reviews > 0 else 0

        if len(dates) > 1:
            intervals = dates.diff().dt.total_seconds() / 86400.0
            avg_interval = intervals.mean()
        else:
            avg_interval = 0

        # 14次元特徴量
        state_features = [
            experience_days,
            total_reviews,
            activity_freq,
            acceptance_rate,
            avg_interval,
            accepted_count,
            total_reviews - accepted_count,
            1 if total_reviews > 10 else 0,
            1 if acceptance_rate > 0.5 else 0,
            1 if activity_freq > 1.0 else 0,
        ]

        action_features = [
            1.0 if total_reviews > 5 else 0.5,
            acceptance_rate,
            1.0 if avg_interval < 30 else 0.5,
            1.0 if total_reviews > 20 else 0.5,
        ]

        features = state_features + action_features

        samples.append({
            'cutoff_date': cutoff_date,
            'reviewer_email': dev_email,
            'features': features,
            'label': label
        })

    return samples


def generate_sliding_window_samples(
    df,
    start_date,
    end_date,
    window_months=3,
    slide_months=1,
    min_history_requests=1
):
    """
    スライディングウィンドウで訓練サンプルを生成

    Args:
        start_date: 開始日
        end_date: 終了日（この日のラベル窓終了まで含む）
        window_months: ラベル窓の長さ（月数）
        slide_months: スライド間隔（月数）
    """
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    all_samples = []
    cutoff_dates = []

    # スライディングウィンドウ（end_dtのラベル窓が含まれるまで）
    current_cutoff = start_dt
    while current_cutoff <= end_dt:
        cutoff_dates.append(current_cutoff)
        current_cutoff += pd.DateOffset(months=slide_months)

    logger.info(f"  スライディングウィンドウ: {len(cutoff_dates)}時点")

    for cutoff_dt in cutoff_dates:
        samples = extract_samples_at_cutoff(
            df,
            cutoff_date=cutoff_dt,
            history_months=12,
            future_months=window_months,
            extended_months=12,
            min_history_requests=min_history_requests
        )

        all_samples.extend(samples)

    logger.info(f"  総訓練サンプル数: {len(all_samples)}")

    return all_samples


def extract_eval_samples(df, eval_start, eval_end, min_history_requests=1):
    """評価サンプル抽出（IRLと同じロジック）"""
    eval_start_dt = pd.to_datetime(eval_start)
    eval_end_dt = pd.to_datetime(eval_end)

    # 履歴期間: 評価開始日の12ヶ月前
    history_start = eval_start_dt - pd.DateOffset(months=12)
    history_end = eval_start_dt

    # 拡張期間
    extended_end = eval_start_dt + pd.DateOffset(months=12)

    history_mask = (df['timestamp'] >= history_start) & (df['timestamp'] < history_end)
    history_data = df[history_mask].copy()

    label_mask = (df['timestamp'] >= eval_start_dt) & (df['timestamp'] < eval_end_dt)
    label_data = df[label_mask].copy()

    extended_mask = (df['timestamp'] >= eval_start_dt) & (df['timestamp'] < extended_end)
    extended_data = df[extended_mask].copy()

    developers = history_data['reviewer_email'].unique()

    samples = []

    for dev_email in developers:
        dev_history = history_data[history_data['reviewer_email'] == dev_email]

        if len(dev_history) < min_history_requests:
            continue

        dev_label = label_data[label_data['reviewer_email'] == dev_email]

        if len(dev_label) == 0:
            dev_extended = extended_data[extended_data['reviewer_email'] == dev_email]
            if len(dev_extended) == 0:
                continue
            label = 0
        else:
            accepted = dev_label[dev_label['label'] == 1]
            label = 1 if len(accepted) > 0 else 0

        # 特徴量計算
        total_reviews = len(dev_history)
        dates = dev_history['timestamp'].sort_values()
        experience_days = (history_end - dates.iloc[0]).days if len(dates) > 0 else 0

        if experience_days > 0:
            activity_freq = total_reviews / (experience_days / 30.0)
        else:
            activity_freq = 0

        accepted_count = (dev_history['label'] == 1).sum()
        acceptance_rate = accepted_count / total_reviews if total_reviews > 0 else 0

        if len(dates) > 1:
            intervals = dates.diff().dt.total_seconds() / 86400.0
            avg_interval = intervals.mean()
        else:
            avg_interval = 0

        state_features = [
            experience_days,
            total_reviews,
            activity_freq,
            acceptance_rate,
            avg_interval,
            accepted_count,
            total_reviews - accepted_count,
            1 if total_reviews > 10 else 0,
            1 if acceptance_rate > 0.5 else 0,
            1 if activity_freq > 1.0 else 0,
        ]

        action_features = [
            1.0 if total_reviews > 5 else 0.5,
            acceptance_rate,
            1.0 if avg_interval < 30 else 0.5,
            1.0 if total_reviews > 20 else 0.5,
        ]

        features = state_features + action_features

        samples.append({
            'reviewer_email': dev_email,
            'features': features,
            'label': label
        })

    return samples


def evaluate_pattern_case1(df, train_end_date, eval_start, eval_end, pattern_name, min_history_requests=1):
    """
    案1: スライディングウィンドウで訓練、2023年評価

    Args:
        train_end_date: 訓練スライド終了日（この日のラベル窓終了まで含む）
        eval_start: 評価開始日
        eval_end: 評価終了日
        pattern_name: パターン名（例: "0-3m → 0-3m"）
    """
    # 訓練データ生成 (2019-01-01から指定終了日までスライド)
    logger.info(f"訓練データ生成: 2019-01-01 ～ {train_end_date} (スライディング)")
    train_samples = generate_sliding_window_samples(
        df,
        start_date='2019-01-01',
        end_date=train_end_date,
        window_months=3,
        slide_months=1,  # 月次スライド
        min_history_requests=min_history_requests
    )

    # 評価データ生成
    logger.info(f"評価データ抽出: {eval_start} ～ {eval_end}")
    eval_samples = extract_eval_samples(df, eval_start, eval_end, min_history_requests)
    logger.info(f"  評価サンプル数: {len(eval_samples)}")

    if len(train_samples) == 0 or len(eval_samples) == 0:
        logger.warning("サンプル数が0のためスキップ")
        return None

    # RF訓練
    X_train = np.array([s['features'] for s in train_samples])
    y_train = np.array([s['label'] for s in train_samples])

    X_eval = np.array([s['features'] for s in eval_samples])
    y_eval = np.array([s['label'] for s in eval_samples])

    logger.info(f"Random Forest訓練（案1）...")
    logger.info(f"  訓練サンプル数: {len(X_train)}")
    logger.info(f"  評価サンプル数: {len(X_eval)}")

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # 評価
    y_pred_proba = rf.predict_proba(X_eval)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_eval, y_pred).ravel()

    metrics = {
        'pattern': pattern_name,
        'train_window': f'2019-01-01 ～ {train_end_date} (sliding)',
        'eval_window': f'{eval_start} ～ {eval_end}',
        'train_samples': len(X_train),
        'eval_samples': len(X_eval),
        'f1': float(f1_score(y_eval, y_pred)),
        'auc_roc': float(roc_auc_score(y_eval, y_pred_proba)),
        'precision': float(precision_score(y_eval, y_pred, zero_division=0)),
        'recall': float(recall_score(y_eval, y_pred, zero_division=0)),
        'accuracy': float(accuracy_score(y_eval, y_pred)),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
    }

    logger.info(f"  F1: {metrics['f1']:.4f}, AUC-ROC: {metrics['auc_roc']:.4f}")

    return metrics


def main():
    logger.info("=" * 80)
    logger.info("RF 案1: スライディングウィンドウ (10パターン版) - Nova単体")
    logger.info("訓練: スライディングウィンドウで生成")
    logger.info("評価: 2023年の各期間")
    logger.info("=" * 80)

    # データ読み込み
    data_path = Path("/Users/kazuki-h/research/multiproject_research/data/review_requests_openstack_multi_5y_detail.csv")
    logger.info(f"データ読み込み: {data_path}")

    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['request_time'])

    # Nova単体フィルタ
    df = df[df['project'] == 'openstack/nova'].copy()
    logger.info(f"Nova単体フィルタ後: {len(df)} records\n")

    # 10パターン定義
    # 訓練終了日: 2021年の各期間終了日に合わせる
    # 評価期間: 2023年の各期間
    patterns = [
        # 0-3m訓練 → 各評価期間
        ("2021-04-01", "2023-01-01", "2023-04-01", "0-3m → 0-3m"),
        ("2021-04-01", "2023-04-01", "2023-07-01", "0-3m → 3-6m"),
        ("2021-04-01", "2023-07-01", "2023-10-01", "0-3m → 6-9m"),
        ("2021-04-01", "2023-10-01", "2024-01-01", "0-3m → 9-12m"),
        # 3-6m訓練 → 同期間以降の評価
        ("2021-07-01", "2023-04-01", "2023-07-01", "3-6m → 3-6m"),
        ("2021-07-01", "2023-07-01", "2023-10-01", "3-6m → 6-9m"),
        ("2021-07-01", "2023-10-01", "2024-01-01", "3-6m → 9-12m"),
        # 6-9m訓練 → 同期間以降の評価
        ("2021-10-01", "2023-07-01", "2023-10-01", "6-9m → 6-9m"),
        ("2021-10-01", "2023-10-01", "2024-01-01", "6-9m → 9-12m"),
        # 9-12m訓練 → 同期間評価
        ("2022-01-01", "2023-10-01", "2024-01-01", "9-12m → 9-12m"),
    ]

    results = []

    for i, (train_end, eval_start, eval_end, pattern_name) in enumerate(patterns, 1):
        logger.info("\n" + "=" * 80)
        logger.info(f"パターン {i}/10: {pattern_name}")
        logger.info("=" * 80)

        metrics = evaluate_pattern_case1(
            df,
            train_end_date=train_end,
            eval_start=eval_start,
            eval_end=eval_end,
            pattern_name=pattern_name,
            min_history_requests=1
        )

        if metrics:
            results.append(metrics)

    # 結果保存
    output_dir = Path("/Users/kazuki-h/research/multiproject_research/outputs/rf_nova_case1_sliding_10patterns")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n結果保存: {output_file}")

    # CSVマトリクス保存
    train_periods = ['0-3m', '3-6m', '6-9m', '9-12m']
    eval_periods = ['0-3m', '3-6m', '6-9m', '9-12m']

    for metric in ['f1', 'auc_roc', 'precision', 'recall']:
        matrix = []
        for train_p in train_periods:
            row = [train_p]
            for eval_p in eval_periods:
                pattern_name = f"{train_p} → {eval_p}"
                matching = [r for r in results if r['pattern'] == pattern_name]
                if matching:
                    row.append(matching[0][metric])
                else:
                    row.append('')
            matrix.append(row)

        matrix_df = pd.DataFrame(matrix, columns=[''] + eval_periods)
        matrix_path = output_dir / f"matrix_{metric}.csv"
        matrix_df.to_csv(matrix_path, index=False)
        logger.info(f"✓ マトリクス保存: {matrix_path}")

    # サマリー
    logger.info("\n" + "=" * 80)
    logger.info("案1: スライディングウィンドウ (10パターン) - サマリー")
    logger.info("=" * 80)

    avg_f1 = np.mean([r['f1'] for r in results])
    avg_auc = np.mean([r['auc_roc'] for r in results])
    avg_recall = np.mean([r['recall'] for r in results])
    avg_precision = np.mean([r['precision'] for r in results])
    avg_train_samples = np.mean([r['train_samples'] for r in results])
    avg_eval_samples = np.mean([r['eval_samples'] for r in results])

    logger.info(f"平均訓練サンプル数: {avg_train_samples:.0f}")
    logger.info(f"平均評価サンプル数: {avg_eval_samples:.0f}")
    logger.info(f"平均 F1:        {avg_f1:.4f}")
    logger.info(f"平均 AUC-ROC:   {avg_auc:.4f}")
    logger.info(f"平均 Recall:    {avg_recall:.4f}")
    logger.info(f"平均 Precision: {avg_precision:.4f}")
    logger.info("=" * 80)

    logger.info("\n完了！")


if __name__ == "__main__":
    main()

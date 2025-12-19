"""
Random Forest Nova単体評価 - 案2: シンプルベースライン

訓練データ: 単一期間から生成 (60-80サンプル)
評価データ: IRLと完全一致 (2023年)
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


def extract_features_simple(
    df,
    history_start,
    history_end,
    label_start,
    label_end,
    extended_label_months=12,
    min_history_requests=1
):
    """
    シンプルな特徴量抽出（案2用）

    1回の時点からサンプルを生成
    """
    history_start_dt = pd.to_datetime(history_start)
    history_end_dt = pd.to_datetime(history_end)
    label_start_dt = pd.to_datetime(label_start)
    label_end_dt = pd.to_datetime(label_end)

    logger.info(f"  履歴期間: {history_start} ～ {history_end}")
    logger.info(f"  ラベル期間: {label_start} ～ {label_end}")

    # 履歴期間のデータ
    history_mask = (df['timestamp'] >= history_start_dt) & (df['timestamp'] < history_end_dt)
    history_data = df[history_mask].copy()

    # ラベル期間のデータ
    label_mask = (df['timestamp'] >= label_start_dt) & (df['timestamp'] < label_end_dt)
    label_data = df[label_mask].copy()

    # 拡張期間のデータ
    extended_label_end_dt = label_start_dt + pd.DateOffset(months=extended_label_months)
    extended_mask = (df['timestamp'] >= label_start_dt) & (df['timestamp'] < extended_label_end_dt)
    extended_data = df[extended_mask].copy()

    logger.info(f"  履歴期間レコード数: {len(history_data)}")
    logger.info(f"  ラベル期間レコード数: {len(label_data)}")
    logger.info(f"  拡張期間レコード数: {len(extended_data)}")

    # 履歴期間に活動した開発者
    developers = history_data['reviewer_email'].unique()
    logger.info(f"  履歴期間の開発者数: {len(developers)}")

    features_list = []
    excluded_min_requests = 0
    excluded_no_extended = 0
    positive_count = 0
    negative_count = 0

    for dev_email in developers:
        dev_history = history_data[history_data['reviewer_email'] == dev_email]

        # 最小依頼数フィルタ
        if len(dev_history) < min_history_requests:
            excluded_min_requests += 1
            continue

        dev_label = label_data[label_data['reviewer_email'] == dev_email]

        # ラベル計算（IRLと同じロジック）
        if len(dev_label) == 0:
            # ラベル期間に依頼なし → 拡張期間をチェック
            dev_extended = extended_data[extended_data['reviewer_email'] == dev_email]

            if len(dev_extended) == 0:
                # 拡張期間にも依頼なし → 除外
                excluded_no_extended += 1
                continue

            # 拡張期間に依頼あり → 負例
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

        # 特徴量計算（14次元）
        total_reviews = len(dev_history)
        dates = dev_history['timestamp'].sort_values()
        experience_days = (history_end_dt - dates.iloc[0]).days if len(dates) > 0 else 0

        # 活動頻度
        if experience_days > 0:
            activity_freq = total_reviews / (experience_days / 30.0)
        else:
            activity_freq = 0

        # 承諾率
        accepted_count = (dev_history['label'] == 1).sum()
        acceptance_rate = accepted_count / total_reviews if total_reviews > 0 else 0

        # 平均活動間隔
        if len(dates) > 1:
            intervals = dates.diff().dt.total_seconds() / 86400.0
            avg_interval = intervals.mean()
        else:
            avg_interval = 0

        # 状態特徴量 (10次元)
        state_features = [
            experience_days,
            total_reviews,
            activity_freq,
            acceptance_rate,
            avg_interval,
            accepted_count,
            total_reviews - accepted_count,  # 拒否数
            1 if total_reviews > 10 else 0,  # 経験フラグ
            1 if acceptance_rate > 0.5 else 0,  # 高承諾率フラグ
            1 if activity_freq > 1.0 else 0,  # 高頻度フラグ
        ]

        # 行動特徴量 (4次元) - 簡易版
        action_features = [
            1.0 if total_reviews > 5 else 0.5,  # 強度
            acceptance_rate,  # 協力度
            1.0 if avg_interval < 30 else 0.5,  # 応答速度
            1.0 if total_reviews > 20 else 0.5,  # 規模
        ]

        features = state_features + action_features

        features_list.append({
            'reviewer_email': dev_email,
            'features': features,
            'label': label
        })

    logger.info(f"  除外（最小依頼数 < {min_history_requests}）: {excluded_min_requests}")
    logger.info(f"  除外（拡張期間にも依頼なし）: {excluded_no_extended}")
    logger.info(f"  正例（継続）: {positive_count} ({positive_count/(positive_count+negative_count)*100:.1f}%)")
    logger.info(f"  負例（離脱）: {negative_count} ({negative_count/(positive_count+negative_count)*100:.1f}%)")
    logger.info(f"  最終サンプル数: {len(features_list)}")

    return pd.DataFrame(features_list)


def evaluate_10_patterns_case2(df, min_history_requests=1):
    """
    案2: シンプルベースライン

    訓練期間: 2021年の各窓から1回だけサンプル生成
    評価期間: 2023年（IRLと一致）
    """
    patterns = [
        {"train": ("2021-01-01", "2021-04-01"), "eval": ("2023-01-01", "2023-04-01"), "name": "0-3m → 0-3m"},
        {"train": ("2021-01-01", "2021-04-01"), "eval": ("2023-04-01", "2023-07-01"), "name": "0-3m → 3-6m"},
        {"train": ("2021-01-01", "2021-04-01"), "eval": ("2023-07-01", "2023-10-01"), "name": "0-3m → 6-9m"},
        {"train": ("2021-01-01", "2021-04-01"), "eval": ("2023-10-01", "2024-01-01"), "name": "0-3m → 9-12m"},
        {"train": ("2021-04-01", "2021-07-01"), "eval": ("2023-04-01", "2023-07-01"), "name": "3-6m → 3-6m"},
        {"train": ("2021-04-01", "2021-07-01"), "eval": ("2023-07-01", "2023-10-01"), "name": "3-6m → 6-9m"},
        {"train": ("2021-04-01", "2021-07-01"), "eval": ("2023-10-01", "2024-01-01"), "name": "3-6m → 9-12m"},
        {"train": ("2021-07-01", "2021-10-01"), "eval": ("2023-07-01", "2023-10-01"), "name": "6-9m → 6-9m"},
        {"train": ("2021-07-01", "2021-10-01"), "eval": ("2023-10-01", "2024-01-01"), "name": "6-9m → 9-12m"},
        {"train": ("2021-10-01", "2022-01-01"), "eval": ("2023-10-01", "2024-01-01"), "name": "9-12m → 9-12m"},
    ]

    results = []

    for i, pattern in enumerate(patterns, 1):
        logger.info("=" * 80)
        logger.info(f"パターン {i}/10: {pattern['name']}")
        logger.info("=" * 80)

        train_start_dt = pd.to_datetime(pattern['train'][0])
        train_end_dt = pd.to_datetime(pattern['train'][1])
        eval_start_dt = pd.to_datetime(pattern['eval'][0])
        eval_end_dt = pd.to_datetime(pattern['eval'][1])

        # 訓練データ: 訓練期間の12ヶ月前から履歴を取る
        train_history_start = train_start_dt - pd.DateOffset(months=12)
        train_history_end = train_start_dt

        logger.info(f"訓練データ抽出（案2: 単一期間）")
        train_features = extract_features_simple(
            df,
            history_start=train_history_start.strftime("%Y-%m-%d"),
            history_end=train_history_end.strftime("%Y-%m-%d"),
            label_start=pattern['train'][0],
            label_end=pattern['train'][1],
            min_history_requests=min_history_requests
        )

        # 評価データ: 評価開始日の12ヶ月前から履歴を取る（仮説2）
        eval_history_start = eval_start_dt - pd.DateOffset(months=12)
        eval_history_end = eval_start_dt

        logger.info(f"評価データ抽出（IRLロジック）")
        eval_features = extract_features_simple(
            df,
            history_start=eval_history_start.strftime("%Y-%m-%d"),
            history_end=eval_history_end.strftime("%Y-%m-%d"),
            label_start=pattern['eval'][0],
            label_end=pattern['eval'][1],
            min_history_requests=min_history_requests
        )

        if len(train_features) == 0 or len(eval_features) == 0:
            logger.warning("サンプル数が0のためスキップ")
            continue

        # RF訓練
        X_train = np.array([f for f in train_features['features']])
        y_train = np.array(train_features['label'])

        X_eval = np.array([f for f in eval_features['features']])
        y_eval = np.array(eval_features['label'])

        logger.info(f"Random Forest訓練...")
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
            'pattern': pattern['name'],
            'train_window': pattern['train'][0] + ' ～ ' + pattern['train'][1],
            'eval_window': pattern['eval'][0] + ' ～ ' + pattern['eval'][1],
            'train_samples': len(X_train),
            'eval_samples': len(X_eval),
            'f1': f1_score(y_eval, y_pred),
            'auc_roc': roc_auc_score(y_eval, y_pred_proba),
            'precision': precision_score(y_eval, y_pred, zero_division=0),
            'recall': recall_score(y_eval, y_pred, zero_division=0),
            'accuracy': accuracy_score(y_eval, y_pred),
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
        }

        logger.info(f"  F1: {metrics['f1']:.4f}, AUC-ROC: {metrics['auc_roc']:.4f}, Recall: {metrics['recall']:.4f}")

        results.append(metrics)

    return results


def main():
    logger.info("=" * 80)
    logger.info("RF 案2: シンプルベースライン - Nova単体")
    logger.info("訓練: 単一期間から生成 (60-80サンプル)")
    logger.info("評価: IRLと完全一致 (2023年)")
    logger.info("=" * 80)

    # データ読み込み
    data_path = Path("/Users/kazuki-h/research/multiproject_research/data/review_requests_openstack_multi_5y_detail.csv")
    logger.info(f"データ読み込み: {data_path}")

    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['request_time'])

    # Nova単体フィルタ
    df = df[df['project'] == 'openstack/nova'].copy()
    logger.info(f"Nova単体フィルタ後: {len(df)} records")

    # 10パターン評価
    results = evaluate_10_patterns_case2(df, min_history_requests=1)

    # 結果保存
    output_dir = Path("/Users/kazuki-h/research/multiproject_research/outputs/rf_nova_case2_simple")
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON保存
    output_file = output_dir / "results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n結果保存: {output_file}")

    # サマリー表示
    logger.info("\n" + "=" * 80)
    logger.info("案2: シンプルベースライン - サマリー")
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

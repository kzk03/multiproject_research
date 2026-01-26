#!/usr/bin/env python3
"""
RF 14次元特徴量重要度抽出（4つの対角線パターン）

IRL対応の14次元特徴量でRFを訓練し、特徴量重要度を抽出
"""

import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 特徴量名（日本語、グラフ用）
FEATURE_NAMES_JP = [
    # 状態特徴量（10次元）
    "経験日数", "総レビュー依頼数", "総レビュー数", "最近の活動頻度",
    "平均活動間隔", "活動トレンド", "協力スコア", "総承諾率",
    "最近の受諾率", "レビュー負荷",
    # 行動特徴量（4次元）
    "応答速度", "協力度", "強度（ファイル数）", "レビュー規模（行数）"
]

# 特徴量名（英語、内部用）
FEATURE_NAMES_EN = [
    "experience_days", "total_changes", "total_reviews", "recent_activity_frequency",
    "avg_activity_gap", "activity_trend", "collaboration_score", "code_quality_score",
    "recent_acceptance_rate", "review_load",
    "avg_action_intensity", "avg_collaboration", "avg_response_time", "avg_review_size"
]


def load_data():
    """Nova単体データを読み込み"""
    data_path = Path("/Users/kazuki-h/research/multiproject_research/data/review_requests_openstack_multi_5y_detail.csv")

    logger.info(f"データ読み込み: {data_path}")
    df = pd.read_csv(data_path)

    # タイムスタンプ列を統一
    if 'request_time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['request_time'])
    elif 'created' in df.columns:
        df['timestamp'] = pd.to_datetime(df['created'])
    else:
        raise ValueError("タイムスタンプ列が見つかりません")

    # email列を統一
    if 'reviewer_email' in df.columns:
        df['email'] = df['reviewer_email']
    elif 'email' not in df.columns:
        df['email'] = df['developer_email']

    # Nova単体にフィルタ
    if 'project' in df.columns:
        df = df[df['project'] == 'openstack/nova'].copy()
        logger.info(f"Nova単体にフィルタ: {len(df)} records")

    return df


def extract_features(df, feature_start, feature_end, label_start, label_end, min_history=3):
    """14次元特徴量を抽出"""
    feature_start_dt = pd.to_datetime(feature_start)
    feature_end_dt = pd.to_datetime(feature_end)
    label_start_dt = pd.to_datetime(label_start)
    label_end_dt = pd.to_datetime(label_end)
    extended_end_dt = label_start_dt + pd.DateOffset(months=12)

    logger.info(f"  特徴量期間: {feature_start} ～ {feature_end}")
    logger.info(f"  ラベル期間: {label_start} ～ {label_end}")

    # データ分割
    feature_mask = (df['timestamp'] >= feature_start_dt) & (df['timestamp'] < feature_end_dt)
    feature_data = df[feature_mask].copy()

    label_mask = (df['timestamp'] >= label_start_dt) & (df['timestamp'] < label_end_dt)
    label_data = df[label_mask].copy()

    extended_mask = (df['timestamp'] >= label_start_dt) & (df['timestamp'] < extended_end_dt)
    extended_data = df[extended_mask].copy()

    developers = feature_data['email'].unique()
    logger.info(f"  開発者数: {len(developers)}")

    features_list = []

    for dev_email in developers:
        dev_feature = feature_data[feature_data['email'] == dev_email]

        if len(dev_feature) < min_history:
            continue

        dev_label = label_data[label_data['email'] == dev_email]

        # ラベル計算
        if len(dev_label) == 0:
            dev_extended = extended_data[extended_data['email'] == dev_email]
            if len(dev_extended) == 0:
                continue  # 拡張期間にも依頼なし → 除外
            label = 0
        else:
            accepted = dev_label[dev_label['label'] == 1]
            label = 1 if len(accepted) > 0 else 0

        # 14次元特徴量計算
        total_reviews = len(dev_feature)
        dates = dev_feature['timestamp'].sort_values()
        experience_days = (feature_end_dt - dates.iloc[0]).days if len(dates) > 0 else 0

        # 活動頻度と間隔
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

        # 14次元ベクトル
        feature_vector = [
            min(experience_days / 730.0, 1.0),  # experience_days
            min(total_reviews / 500.0, 1.0),    # total_changes
            min(total_reviews / 500.0, 1.0),    # total_reviews
            min(activity_freq, 1.0),            # recent_activity_frequency
            min(avg_gap / 60.0, 1.0),           # avg_activity_gap
            (trend + 1) / 2,                    # activity_trend (0-1に正規化)
            min(collab_score, 1.0),             # collaboration_score
            min(acceptance_rate, 1.0),          # code_quality_score
            min(acceptance_rate, 1.0),          # recent_acceptance_rate
            min(total_reviews / 90, 1.0),       # review_load
            min(activity_freq, 1.0),            # avg_action_intensity
            min(collab_score, 1.0),             # avg_collaboration
            min(avg_gap / 60.0, 1.0),           # avg_response_time
            min(review_size, 1.0),              # avg_review_size
        ]

        features_list.append({
            'email': dev_email,
            'features': feature_vector,
            'label': label
        })

    logger.info(f"  抽出サンプル数: {len(features_list)}")

    if len(features_list) == 0:
        return None, None

    X = np.array([f['features'] for f in features_list])
    y = np.array([f['label'] for f in features_list])

    logger.info(f"  継続者: {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.1f}%)")
    logger.info(f"  離脱者: {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.1f}%)")

    return X, y


def train_and_extract_importance(pattern_name, X_train, y_train):
    """RFを訓練して特徴量重要度を取得"""
    logger.info(f"\n{pattern_name} でRF訓練中...")

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    importances = rf.feature_importances_

    logger.info(f"特徴量重要度:")
    importance_dict = {}
    for i, (name_jp, importance) in enumerate(zip(FEATURE_NAMES_JP, importances)):
        logger.info(f"  {name_jp:20s}: {importance:.4f}")
        importance_dict[name_jp] = float(importance)

    return importance_dict


def main():
    logger.info("=" * 80)
    logger.info("RF 14次元特徴量重要度抽出（4つの対角線パターン）")
    logger.info("=" * 80)

    # データ読み込み
    df = load_data()

    # 4つの対角線パターン
    patterns = [
        {"name": "0-3m", "feature": ("2021-01-01", "2021-04-01"), "label": ("2021-04-01", "2021-07-01")},
        {"name": "3-6m", "feature": ("2021-04-01", "2021-07-01"), "label": ("2021-07-01", "2021-10-01")},
        {"name": "6-9m", "feature": ("2021-07-01", "2021-10-01"), "label": ("2021-10-01", "2022-01-01")},
        {"name": "9-12m", "feature": ("2021-10-01", "2022-01-01"), "label": ("2022-01-01", "2022-04-01")},
    ]

    all_results = []

    for pattern in patterns:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"パターン: {pattern['name']} → {pattern['name']}")
        logger.info(f"{'=' * 80}")

        X, y = extract_features(
            df,
            pattern['feature'][0],
            pattern['feature'][1],
            pattern['label'][0],
            pattern['label'][1]
        )

        if X is None or len(X) == 0:
            logger.warning(f"  サンプルなし、スキップ")
            continue

        importance_dict = train_and_extract_importance(f"{pattern['name']} → {pattern['name']}", X, y)

        all_results.append({
            'pattern': f"{pattern['name']} → {pattern['name']}",
            'feature_importances': importance_dict
        })

    # 結果保存
    output_dir = Path("/Users/kazuki-h/research/multiproject_research/outputs/singleproject/rf_nova_14dim")
    output_dir.mkdir(parents=True, exist_ok=True)

    importance_dir = output_dir / "feature_importance"
    importance_dir.mkdir(parents=True, exist_ok=True)

    # JSON保存
    json_file = importance_dir / "all_patterns_14dim.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"\n✓ 特徴量重要度保存: {json_file}")

    # CSV保存（サマリー）
    df_summary = pd.DataFrame([
        {'pattern': r['pattern'], **r['feature_importances']}
        for r in all_results
    ])
    csv_file = importance_dir / "feature_importance_summary_14dim.csv"
    df_summary.to_csv(csv_file, index=False)
    logger.info(f"✓ サマリーCSV保存: {csv_file}")

    # 平均重要度
    avg_importance = df_summary[FEATURE_NAMES_JP].mean().sort_values(ascending=False)
    avg_file = importance_dir / "average_importance_14dim.csv"
    avg_importance.to_csv(avg_file, header=['importance'])
    logger.info(f"✓ 平均重要度保存: {avg_file}")

    logger.info("\n" + "=" * 80)
    logger.info("完了！")
    logger.info("=" * 80)

    logger.info("\n平均特徴量重要度（Top 14）:")
    for i, (feat, imp) in enumerate(avg_importance.items(), 1):
        logger.info(f"  {i:2d}. {feat:20s}: {imp:.4f}")


if __name__ == '__main__':
    main()

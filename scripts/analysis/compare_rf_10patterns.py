#!/usr/bin/env python3
"""
Random Forest 10パターン評価スクリプト

IRLと同じ10パターン（train_window → eval_window）でRandom Forestを評価
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    precision_score, recall_score, accuracy_score,
    confusion_matrix
)
import json
import time

def load_features(train_window: str, eval_window: str):
    """特徴量を読み込み"""
    base_dir = Path("/Users/kazuki-h/research/multiproject_research/outputs/analysis_data/irl_timeseries_vs_rf_final")

    train_path = base_dir / f"train_{train_window}_features.csv"
    eval_path = base_dir / f"eval_{eval_window}_features.csv"

    if not train_path.exists() or not eval_path.exists():
        return None, None

    train_df = pd.read_csv(train_path)
    eval_df = pd.read_csv(eval_path)

    return train_df, eval_df


def prepare_features(df: pd.DataFrame):
    """特徴量を準備（IRLと同じ特徴量セット）"""
    # 状態特徴量（14次元）
    state_features = [
        'experience_days', 'total_changes', 'total_reviews',
        'recent_activity_frequency', 'avg_activity_gap', 'activity_trend',
        'collaboration_score', 'code_quality_score', 'recent_acceptance_rate',
        'review_load', 'project_count', 'project_activity_distribution',
        'main_project_contribution_ratio', 'cross_project_collaboration_score'
    ]

    # 行動特徴量（5次元）
    action_features = [
        'avg_action_intensity', 'avg_collaboration',
        'avg_response_time', 'avg_review_size',
        'cross_project_action_ratio'
    ]

    all_features = state_features + action_features

    # 存在する特徴量のみ選択
    available_features = [f for f in all_features if f in df.columns]

    # activity_trendを数値化
    if 'activity_trend' in df.columns:
        trend_mapping = {'increasing': 1.0, 'stable': 0.0, 'decreasing': -1.0}
        df = df.copy()
        df['activity_trend'] = df['activity_trend'].map(trend_mapping).fillna(0)

    X = df[available_features].fillna(0)
    y = df['label'] if 'label' in df.columns else None

    return X, y, available_features


def train_and_evaluate_rf(train_df, eval_df, pattern_name):
    """Random Forestを訓練・評価"""
    print(f"\n{'='*60}")
    print(f"パターン: {pattern_name}")
    print(f"{'='*60}")

    # 特徴量準備
    X_train, y_train, features = prepare_features(train_df)
    X_eval, y_eval, _ = prepare_features(eval_df)

    print(f"訓練サンプル数: {len(X_train)} (Positive: {y_train.sum()}, Negative: {len(y_train) - y_train.sum()})")
    print(f"評価サンプル数: {len(X_eval)} (Positive: {y_eval.sum()}, Negative: {len(y_eval) - y_eval.sum()})")
    print(f"特徴量数: {len(features)}")

    # Random Forest訓練
    start_time = time.time()
    rf = RandomForestClassifier(
        n_estimators=100,
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

    # メトリクス計算
    f1 = f1_score(y_eval, y_pred)
    auc_roc = roc_auc_score(y_eval, y_pred_proba)
    auc_pr = average_precision_score(y_eval, y_pred_proba)
    precision = precision_score(y_eval, y_pred, zero_division=0)
    recall = recall_score(y_eval, y_pred, zero_division=0)
    accuracy = accuracy_score(y_eval, y_pred)

    # 混同行列
    tn, fp, fn, tp = confusion_matrix(y_eval, y_pred).ravel()

    # 結果表示
    print(f"\n結果:")
    print(f"  F1 Score:   {f1:.4f}")
    print(f"  AUC-ROC:    {auc_roc:.4f}")
    print(f"  AUC-PR:     {auc_pr:.4f}")
    print(f"  Precision:  {precision:.4f}")
    print(f"  Recall:     {recall:.4f}")
    print(f"  Accuracy:   {accuracy:.4f}")
    print(f"  TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"  訓練時間: {train_time:.4f}秒")
    print(f"  予測時間: {predict_time:.4f}秒")

    return {
        'pattern': pattern_name,
        'train_window': pattern_name.split(' → ')[0],
        'eval_window': pattern_name.split(' → ')[1],
        'f1': f1,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'train_time': train_time,
        'predict_time': predict_time,
        'train_samples': len(X_train),
        'eval_samples': len(X_eval),
        'n_features': len(features)
    }


def main():
    print("=" * 60)
    print("Random Forest 10パターン評価")
    print("=" * 60)

    # 10パターン定義
    patterns = [
        ("0-3m", "0-3m"),
        ("0-3m", "3-6m"),
        ("0-3m", "6-9m"),
        ("0-3m", "9-12m"),
        ("3-6m", "3-6m"),
        ("3-6m", "6-9m"),
        ("3-6m", "9-12m"),
        ("6-9m", "6-9m"),
        ("6-9m", "9-12m"),
        ("9-12m", "9-12m")
    ]

    results = []

    for train_window, eval_window in patterns:
        pattern_name = f"{train_window} → {eval_window}"

        # 特徴量ファイル確認
        train_df, eval_df = load_features(train_window, eval_window)

        if train_df is None or eval_df is None:
            print(f"\n⚠️ {pattern_name}: 特徴量ファイルが見つかりません")
            continue

        # RF訓練・評価
        result = train_and_evaluate_rf(train_df, eval_df, pattern_name)
        results.append(result)

    # 結果保存
    output_dir = Path("/Users/kazuki-h/research/multiproject_research/outputs/analysis_data/rf_10patterns")
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV保存
    results_df = pd.DataFrame(results)
    csv_path = output_dir / "rf_10patterns_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n✓ 結果をCSVに保存: {csv_path}")

    # JSON保存
    json_path = output_dir / "rf_10patterns_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ 結果をJSONに保存: {json_path}")

    # サマリー表示
    print("\n" + "=" * 60)
    print("全パターン サマリー")
    print("=" * 60)

    avg_f1 = results_df['f1'].mean()
    avg_auc_roc = results_df['auc_roc'].mean()
    avg_recall = results_df['recall'].mean()
    avg_precision = results_df['precision'].mean()

    print(f"\n平均メトリクス（{len(results)}パターン）:")
    print(f"  平均 F1:        {avg_f1:.4f}")
    print(f"  平均 AUC-ROC:   {avg_auc_roc:.4f}")
    print(f"  平均 Recall:    {avg_recall:.4f}")
    print(f"  平均 Precision: {avg_precision:.4f}")

    print(f"\nF1スコア範囲:")
    print(f"  最高: {results_df['f1'].max():.4f} ({results_df.loc[results_df['f1'].idxmax(), 'pattern']})")
    print(f"  最低: {results_df['f1'].min():.4f} ({results_df.loc[results_df['f1'].idxmin(), 'pattern']})")

    print("\n" + "=" * 60)
    print("完了")
    print("=" * 60)


if __name__ == "__main__":
    main()

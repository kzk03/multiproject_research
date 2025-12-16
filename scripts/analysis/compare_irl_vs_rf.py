#!/usr/bin/env python3
"""
IRL vs Random Forest 比較実験

IRLモデルとRandom Forestの予測性能を比較する。
- 全体性能比較（F1, AUC-ROC, Precision, Recall）
- プロジェクトタイプ別比較（Expert/Contributor/Specialist）
- 計算コスト比較（訓練時間、予測時間、メモリ）
"""

import json
import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_features_and_predictions(features_path: Path, predictions_path: Path):
    """
    特徴量とIRL予測結果を読み込み
    """
    logger.info(f"特徴量を読み込み: {features_path}")
    df_features = pd.read_csv(features_path)

    logger.info(f"IRL予測結果を読み込み: {predictions_path}")
    df_predictions = pd.read_csv(predictions_path)

    # マージ
    df = pd.merge(
        df_features,
        df_predictions[['reviewer_email', 'predicted_prob', 'predicted_binary']],
        on='reviewer_email',
        how='inner',
        suffixes=('', '_irl')
    )

    logger.info(f"マージ後のレコード数: {len(df)}")
    return df


def prepare_features(df: pd.DataFrame):
    """
    特徴量を準備（IRLと同じ19次元）
    """
    # 14次元状態特徴量
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
        'project_count',
        'project_activity_distribution',
        'main_project_contribution_ratio',
        'cross_project_collaboration_score',
    ]

    # 5次元行動特徴量
    action_features = [
        'avg_action_intensity',
        'avg_collaboration',
        'avg_response_time',
        'avg_review_size',
        'cross_project_action_ratio',
    ]

    all_features = state_features + action_features

    # activity_trendを数値に変換
    trend_mapping = {
        'increasing': 1.0,
        'stable': 0.0,
        'decreasing': -1.0
    }
    if 'activity_trend' in df.columns:
        df['activity_trend'] = df['activity_trend'].map(trend_mapping).fillna(0)

    X = df[all_features].fillna(0)
    y = df['true_label']

    return X, y, all_features


def train_random_forest(X: np.ndarray, y: np.ndarray, config: dict):
    """
    Random Forestを訓練
    """
    logger.info(f"Random Forestを訓練: {config['name']}")

    rf = RandomForestClassifier(
        n_estimators=config.get('n_estimators', 200),
        max_depth=config.get('max_depth', 20),
        min_samples_split=config.get('min_samples_split', 5),
        min_samples_leaf=config.get('min_samples_leaf', 2),
        class_weight=config.get('class_weight', 'balanced'),
        random_state=42,
        n_jobs=-1
    )

    start_time = time.time()
    rf.fit(X, y)
    train_time = time.time() - start_time

    logger.info(f"訓練完了: {train_time:.2f}秒")

    return rf, train_time


def train_baseline_models(X: np.ndarray, y: np.ndarray):
    """
    ベースラインモデル（Logistic Regression）を訓練
    """
    logger.info("Logistic Regressionを訓練")

    lr = LogisticRegression(
        class_weight='balanced',
        random_state=42,
        max_iter=1000
    )

    start_time = time.time()
    lr.fit(X, y)
    train_time = time.time() - start_time

    logger.info(f"訓練完了: {train_time:.2f}秒")

    return lr, train_time


def evaluate_model(model, X: np.ndarray, y: np.ndarray, model_name: str):
    """
    モデルを評価
    """
    logger.info(f"{model_name}を評価中...")

    # 予測
    start_time = time.time()
    y_pred_proba = model.predict_proba(X)[:, 1]
    predict_time = time.time() - start_time

    y_pred = (y_pred_proba > 0.5).astype(int)

    # 評価指標
    f1 = f1_score(y, y_pred)
    auc_roc = roc_auc_score(y, y_pred_proba)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    accuracy = accuracy_score(y, y_pred)

    # Precision-Recall AUC
    precision_vals, recall_vals, _ = precision_recall_curve(y, y_pred_proba)
    auc_pr = auc(recall_vals, precision_vals)

    # 混同行列
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    results = {
        'model': model_name,
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
        'predict_time': predict_time,
        'predictions': y_pred_proba
    }

    logger.info(f"{model_name} - F1: {f1:.4f}, AUC-ROC: {auc_roc:.4f}, Recall: {recall:.4f}")

    return results


def evaluate_irl_model(df: pd.DataFrame):
    """
    IRL予測結果を評価
    """
    logger.info("IRLモデルを評価中...")

    y_true = df['true_label'].values
    y_pred_proba = df['predicted_prob'].values
    y_pred = (y_pred_proba > 0.5).astype(int)

    # 評価指標
    f1 = f1_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_pred_proba)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    # Precision-Recall AUC
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_proba)
    auc_pr = auc(recall_vals, precision_vals)

    # 混同行列
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    results = {
        'model': 'IRL',
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
        'predict_time': 0.0,  # 既に予測済み
        'predictions': y_pred_proba
    }

    logger.info(f"IRL - F1: {f1:.4f}, AUC-ROC: {auc_roc:.4f}, Recall: {recall:.4f}")

    return results


def classify_project_type(count):
    """プロジェクト数からタイプを分類"""
    if count == 1:
        return 'Specialist (1 proj)'
    elif count <= 3:
        return 'Contributor (2-3 proj)'
    else:
        return 'Expert (4+ proj)'


def evaluate_by_project_type(df: pd.DataFrame, predictions_dict: dict, output_dir: Path):
    """
    プロジェクトタイプ別の精度を評価
    """
    logger.info("=" * 80)
    logger.info("プロジェクトタイプ別評価")
    logger.info("=" * 80)

    df['project_type'] = df['project_count'].apply(classify_project_type)

    results = []

    for model_name, pred_proba in predictions_dict.items():
        df[f'pred_{model_name}'] = (pred_proba > 0.5).astype(int)

        for proj_type in ['Expert (4+ proj)', 'Contributor (2-3 proj)', 'Specialist (1 proj)']:
            subset = df[df['project_type'] == proj_type]
            if len(subset) == 0:
                continue

            y_true = subset['true_label']
            y_pred = subset[f'pred_{model_name}']

            accuracy = (y_true == y_pred).mean()
            count = len(subset)

            results.append({
                'model': model_name,
                'project_type': proj_type,
                'accuracy': accuracy,
                'count': count
            })

            logger.info(f"{model_name} - {proj_type}: {accuracy:.3f} (N={count})")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'project_type_comparison.csv', index=False)

    # 可視化
    fig, ax = plt.subplots(figsize=(12, 6))

    project_types = ['Expert (4+ proj)', 'Contributor (2-3 proj)', 'Specialist (1 proj)']
    x = np.arange(len(project_types))
    width = 0.25

    models = list(predictions_dict.keys())
    for i, model in enumerate(models):
        model_data = results_df[results_df['model'] == model]
        accuracies = [
            model_data[model_data['project_type'] == pt]['accuracy'].values[0]
            if len(model_data[model_data['project_type'] == pt]) > 0 else 0
            for pt in project_types
        ]
        ax.bar(x + i * width, accuracies, width, label=model, alpha=0.8)

    ax.set_xlabel('Project Type', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Prediction Accuracy by Project Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(project_types, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(output_dir / 'project_type_comparison.png', dpi=300, bbox_inches='tight')
    logger.info(f"可視化を保存: {output_dir / 'project_type_comparison.png'}")
    plt.close()


def create_comparison_visualizations(all_results: list, df: pd.DataFrame, output_dir: Path):
    """
    比較可視化を作成
    """
    logger.info("=" * 80)
    logger.info("比較可視化を作成")
    logger.info("=" * 80)

    # (1) 性能比較棒グラフ
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    metrics = ['f1', 'auc_roc', 'auc_pr', 'precision', 'recall', 'accuracy']
    metric_names = ['F1 Score', 'AUC-ROC', 'AUC-PR', 'Precision', 'Recall', 'Accuracy']

    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        models = [r['model'] for r in all_results]
        values = [r[metric] for r in all_results]

        colors = ['#1f77b4' if m == 'IRL' else '#ff7f0e' if 'RF' in m else '#2ca02c' for m in models]
        axes[i].bar(range(len(models)), values, color=colors, alpha=0.7)
        axes[i].set_xticks(range(len(models)))
        axes[i].set_xticklabels(models, rotation=45, ha='right')
        axes[i].set_ylabel(name, fontsize=11)
        axes[i].set_title(f'{name} Comparison', fontsize=12, fontweight='bold')
        axes[i].set_ylim(0, 1.0)
        axes[i].grid(axis='y', alpha=0.3)

        # 最高値に星印
        max_idx = np.argmax(values)
        axes[i].text(max_idx, values[max_idx] + 0.02, '★', ha='center', fontsize=20, color='red')

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    logger.info(f"可視化を保存: {output_dir / 'performance_comparison.png'}")
    plt.close()

    # (2) ROC曲線比較
    fig, ax = plt.subplots(figsize=(10, 8))

    y_true = df['true_label'].values

    for result in all_results:
        y_pred_proba = result['predictions']
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = result['auc_roc']

        ax.plot(fpr, tpr, label=f"{result['model']} (AUC={auc_score:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'roc_comparison.png', dpi=300, bbox_inches='tight')
    logger.info(f"可視化を保存: {output_dir / 'roc_comparison.png'}")
    plt.close()

    # (3) Precision-Recall曲線比較
    fig, ax = plt.subplots(figsize=(10, 8))

    for result in all_results:
        y_pred_proba = result['predictions']
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_proba)
        auc_score = result['auc_pr']

        ax.plot(recall_vals, precision_vals, label=f"{result['model']} (AUC={auc_score:.3f})", linewidth=2)

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'pr_comparison.png', dpi=300, bbox_inches='tight')
    logger.info(f"可視化を保存: {output_dir / 'pr_comparison.png'}")
    plt.close()

    # (4) レーダーチャート
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    categories = ['F1', 'AUC-ROC', 'AUC-PR', 'Precision', 'Recall', 'Accuracy']
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    for result in all_results:
        values = [result['f1'], result['auc_roc'], result['auc_pr'],
                  result['precision'], result['recall'], result['accuracy']]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=result['model'])
        ax.fill(angles, values, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_title('Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'radar_comparison.png', dpi=300, bbox_inches='tight')
    logger.info(f"可視化を保存: {output_dir / 'radar_comparison.png'}")
    plt.close()


def create_summary_report(all_results: list, train_times: dict, output_dir: Path):
    """
    サマリーレポートを作成
    """
    logger.info("=" * 80)
    logger.info("サマリーレポート作成")
    logger.info("=" * 80)

    # CSV保存
    results_df = pd.DataFrame([
        {
            'model': r['model'],
            'f1': r['f1'],
            'auc_roc': r['auc_roc'],
            'auc_pr': r['auc_pr'],
            'precision': r['precision'],
            'recall': r['recall'],
            'accuracy': r['accuracy'],
            'tp': r['tp'],
            'tn': r['tn'],
            'fp': r['fp'],
            'fn': r['fn'],
            'train_time': train_times.get(r['model'], 0.0),
            'predict_time': r['predict_time']
        }
        for r in all_results
    ])

    results_df.to_csv(output_dir / 'model_comparison_summary.csv', index=False)
    logger.info(f"サマリーを保存: {output_dir / 'model_comparison_summary.csv'}")

    # Markdownレポート
    report_lines = [
        "# IRL vs Random Forest 比較実験レポート",
        "",
        "## モデル性能比較",
        "",
        "| Model | F1 | AUC-ROC | AUC-PR | Precision | Recall | Accuracy |",
        "|-------|-----|---------|--------|-----------|--------|----------|"
    ]

    for _, row in results_df.iterrows():
        report_lines.append(
            f"| **{row['model']}** | {row['f1']:.4f} | {row['auc_roc']:.4f} | "
            f"{row['auc_pr']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['accuracy']:.4f} |"
        )

    report_lines.extend([
        "",
        "## 混同行列",
        "",
        "| Model | TP | TN | FP | FN |",
        "|-------|----|----|----|----|"
    ])

    for _, row in results_df.iterrows():
        report_lines.append(
            f"| **{row['model']}** | {row['tp']} | {row['tn']} | {row['fp']} | {row['fn']} |"
        )

    report_lines.extend([
        "",
        "## 計算コスト",
        "",
        "| Model | Train Time (s) | Predict Time (s) |",
        "|-------|---------------|-----------------|"
    ])

    for _, row in results_df.iterrows():
        report_lines.append(
            f"| **{row['model']}** | {row['train_time']:.4f} | {row['predict_time']:.4f} |"
        )

    report_lines.extend([
        "",
        "## 主要発見",
        "",
        f"### 最高F1スコア",
        f"**{results_df.loc[results_df['f1'].idxmax(), 'model']}**: {results_df['f1'].max():.4f}",
        "",
        f"### 最高AUC-ROC",
        f"**{results_df.loc[results_df['auc_roc'].idxmax(), 'model']}**: {results_df['auc_roc'].max():.4f}",
        "",
        f"### 最高Recall",
        f"**{results_df.loc[results_df['recall'].idxmax(), 'model']}**: {results_df['recall'].max():.4f}",
        "",
        "## 可視化",
        "",
        "- [performance_comparison.png](performance_comparison.png) - 性能比較棒グラフ",
        "- [roc_comparison.png](roc_comparison.png) - ROC曲線比較",
        "- [pr_comparison.png](pr_comparison.png) - Precision-Recall曲線比較",
        "- [radar_comparison.png](radar_comparison.png) - レーダーチャート",
        "- [project_type_comparison.png](project_type_comparison.png) - プロジェクトタイプ別精度",
        ""
    ])

    report_path = output_dir / 'comparison_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    logger.info(f"レポートを保存: {report_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='IRL vs Random Forest 比較実験')
    parser.add_argument('--features', required=True, help='特徴量CSVパス')
    parser.add_argument('--predictions', required=True, help='IRL予測結果CSVパス')
    parser.add_argument('--output', required=True, help='出力ディレクトリ')

    args = parser.parse_args()

    # 出力ディレクトリ作成
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # データ読み込み
    df = load_features_and_predictions(Path(args.features), Path(args.predictions))
    X, y, feature_names = prepare_features(df)

    logger.info(f"サンプル数: {len(X)}")
    logger.info(f"特徴量数: {len(feature_names)}")
    logger.info(f"正例: {y.sum()} ({y.mean()*100:.1f}%)")

    # モデル訓練・評価
    all_results = []
    train_times = {}
    predictions_dict = {}

    # (1) IRL評価
    irl_results = evaluate_irl_model(df)
    all_results.append(irl_results)
    predictions_dict['IRL'] = irl_results['predictions']

    # (2) Random Forest
    rf_configs = [
        {'name': 'Random Forest', 'n_estimators': 200, 'max_depth': 20, 'class_weight': 'balanced'},
        {'name': 'Random Forest (Deep)', 'n_estimators': 200, 'max_depth': None, 'class_weight': 'balanced'},
        {'name': 'Random Forest (More Trees)', 'n_estimators': 500, 'max_depth': 20, 'class_weight': 'balanced'},
    ]

    for config in rf_configs:
        rf_model, train_time = train_random_forest(X.values, y.values, config)
        rf_results = evaluate_model(rf_model, X.values, y.values, config['name'])

        all_results.append(rf_results)
        train_times[config['name']] = train_time
        predictions_dict[config['name']] = rf_results['predictions']

    # (3) Logistic Regression（ベースライン）
    lr_model, lr_train_time = train_baseline_models(X.values, y.values)
    lr_results = evaluate_model(lr_model, X.values, y.values, 'Logistic Regression')
    all_results.append(lr_results)
    train_times['Logistic Regression'] = lr_train_time
    predictions_dict['Logistic Regression'] = lr_results['predictions']

    # プロジェクトタイプ別評価
    evaluate_by_project_type(df, predictions_dict, output_dir)

    # 可視化
    create_comparison_visualizations(all_results, df, output_dir)

    # サマリーレポート
    create_summary_report(all_results, train_times, output_dir)

    logger.info("\n" + "=" * 80)
    logger.info("すべての比較実験が完了しました！")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

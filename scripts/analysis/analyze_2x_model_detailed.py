#!/usr/bin/env python3
"""
2x OSモデル（最良性能）の詳細分析

1. 特徴量重要度分析（Permutation Importance）
2. 予測成功した開発者の特性
3. 予測失敗した開発者の特性
4. 特徴量の分布比較
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_feature_importance(df: pd.DataFrame, output_dir: Path):
    """
    特徴量重要度を分析（Permutation Importance）
    """
    logger.info("=" * 80)
    logger.info("特徴量重要度分析")
    logger.info("=" * 80)

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

    # データ準備
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

    logger.info(f"サンプル数: {len(X)}")
    logger.info(f"特徴量数: {len(all_features)}")
    logger.info(f"正例: {y.sum()} ({y.mean()*100:.1f}%)")

    # Random Forestで特徴量重要度を計算
    logger.info("\nRandom Forestで特徴量重要度を計算中...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X, y)

    # Permutation Importance
    logger.info("Permutation Importance計算中...")
    perm_importance = permutation_importance(
        rf, X, y, n_repeats=10, random_state=42, n_jobs=-1
    )

    # 重要度をデータフレームに
    importance_df = pd.DataFrame({
        'feature': all_features,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std,
        'rf_importance': rf.feature_importances_
    }).sort_values('importance_mean', ascending=False)

    # 特徴量タイプを追加
    importance_df['feature_type'] = importance_df['feature'].apply(
        lambda x: 'State (14)' if x in state_features else 'Action (5)'
    )

    # 日本語名を追加
    feature_names_ja = {
        'experience_days': '経験日数',
        'total_changes': '総変更数',
        'total_reviews': '総レビュー数',
        'recent_activity_frequency': '最近の活動頻度',
        'avg_activity_gap': '平均活動間隔',
        'activity_trend': '活動トレンド',
        'collaboration_score': 'コラボレーションスコア',
        'code_quality_score': 'コード品質スコア',
        'recent_acceptance_rate': '最近の承諾率',
        'review_load': 'レビュー負荷',
        'project_count': 'プロジェクト数',
        'project_activity_distribution': 'プロジェクト活動分布',
        'main_project_contribution_ratio': 'メインプロジェクト貢献率',
        'cross_project_collaboration_score': 'クロスプロジェクトコラボレーションスコア',
        'avg_action_intensity': '平均行動強度',
        'avg_collaboration': '平均コラボレーション',
        'avg_response_time': '平均応答時間',
        'avg_review_size': '平均レビューサイズ',
        'cross_project_action_ratio': 'クロスプロジェクト行動比率',
    }
    importance_df['feature_ja'] = importance_df['feature'].map(feature_names_ja)

    # CSV保存
    importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)
    logger.info(f"\n特徴量重要度を保存: {output_dir / 'feature_importance.csv'}")

    # Top 10を表示
    logger.info("\n【Top 10 重要特徴量】")
    for i, row in importance_df.head(10).iterrows():
        logger.info(f"  {row['feature_ja']:30s} {row['importance_mean']:.4f} ± {row['importance_std']:.4f}")

    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # (1) Permutation Importance（Top 15）
    top_features = importance_df.head(15)
    axes[0].barh(range(len(top_features)), top_features['importance_mean'],
                 xerr=top_features['importance_std'], color='steelblue')
    axes[0].set_yticks(range(len(top_features)))
    axes[0].set_yticklabels(top_features['feature_ja'])
    axes[0].set_xlabel('Permutation Importance', fontsize=12)
    axes[0].set_title('Top 15 Feature Importance (Permutation)', fontsize=14, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)

    # (2) 特徴量タイプ別の平均重要度
    type_importance = importance_df.groupby('feature_type')['importance_mean'].agg(['mean', 'std'])
    axes[1].bar(range(len(type_importance)), type_importance['mean'],
                yerr=type_importance['std'], color=['#1f77b4', '#ff7f0e'], alpha=0.7)
    axes[1].set_xticks(range(len(type_importance)))
    axes[1].set_xticklabels(type_importance.index, fontsize=12)
    axes[1].set_ylabel('Average Permutation Importance', fontsize=12)
    axes[1].set_title('Feature Importance by Type', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
    logger.info(f"可視化を保存: {output_dir / 'feature_importance.png'}")
    plt.close()

    return importance_df


def analyze_correct_vs_incorrect_predictions(df: pd.DataFrame, output_dir: Path):
    """
    予測成功 vs 失敗の開発者特性を比較
    """
    logger.info("=" * 80)
    logger.info("予測成功 vs 失敗の開発者特性分析")
    logger.info("=" * 80)

    # 予測結果を分類
    df['pred_binary'] = (df['predicted_prob'] > 0.5).astype(int)
    df['is_correct'] = (df['pred_binary'] == df['true_label']).astype(int)

    correct = df[df['is_correct'] == 1]
    incorrect = df[df['is_correct'] == 0]

    logger.info(f"\n予測成功: {len(correct)} ({len(correct)/len(df)*100:.1f}%)")
    logger.info(f"予測失敗: {len(incorrect)} ({len(incorrect)/len(df)*100:.1f}%)")

    # 特徴量
    features = [
        'experience_days',
        'total_changes',
        'total_reviews',
        'recent_activity_frequency',
        'avg_activity_gap',
        'collaboration_score',
        'code_quality_score',
        'recent_acceptance_rate',
        'review_load',
        'project_count',
        'cross_project_collaboration_score',
        'avg_action_intensity',
    ]

    # 統計比較
    comparison_stats = []
    for feat in features:
        correct_mean = correct[feat].mean()
        correct_std = correct[feat].std()
        incorrect_mean = incorrect[feat].mean()
        incorrect_std = incorrect[feat].std()

        comparison_stats.append({
            'feature': feat,
            'correct_mean': correct_mean,
            'correct_std': correct_std,
            'incorrect_mean': incorrect_mean,
            'incorrect_std': incorrect_std,
            'diff': correct_mean - incorrect_mean,
            'diff_pct': ((correct_mean - incorrect_mean) / (incorrect_mean + 1e-10)) * 100
        })

    stats_df = pd.DataFrame(comparison_stats)
    stats_df.to_csv(output_dir / 'correct_vs_incorrect_stats.csv', index=False)
    logger.info(f"\n統計を保存: {output_dir / 'correct_vs_incorrect_stats.csv'}")

    # 大きな差がある特徴量をログ出力
    logger.info("\n【予測成功 vs 失敗で差が大きい特徴量】")
    for _, row in stats_df.nlargest(5, 'diff_pct').iterrows():
        logger.info(f"  {row['feature']:40s} 成功: {row['correct_mean']:.3f}, 失敗: {row['incorrect_mean']:.3f}, 差: {row['diff_pct']:+.1f}%")

    # 可視化（バイオリンプロット）
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        if i >= len(axes):
            break

        data_to_plot = [
            correct[feat].dropna(),
            incorrect[feat].dropna()
        ]

        parts = axes[i].violinplot(data_to_plot, positions=[0, 1], showmeans=True, showmedians=True)
        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels(['Correct', 'Incorrect'])
        axes[i].set_ylabel(feat, fontsize=10)
        axes[i].set_title(f'{feat}', fontsize=11, fontweight='bold')
        axes[i].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'correct_vs_incorrect_violin.png', dpi=300, bbox_inches='tight')
    logger.info(f"可視化を保存: {output_dir / 'correct_vs_incorrect_violin.png'}")
    plt.close()

    return stats_df


def analyze_false_positive_false_negative(df: pd.DataFrame, output_dir: Path):
    """
    False Positive と False Negative の詳細分析
    """
    logger.info("=" * 80)
    logger.info("False Positive / False Negative 分析")
    logger.info("=" * 80)

    # 予測分類
    df['pred_binary'] = (df['predicted_prob'] > 0.5).astype(int)

    def classify_prediction(row):
        if row['pred_binary'] == 1 and row['true_label'] == 1:
            return 'TP'
        elif row['pred_binary'] == 1 and row['true_label'] == 0:
            return 'FP'
        elif row['pred_binary'] == 0 and row['true_label'] == 1:
            return 'FN'
        else:
            return 'TN'

    df['prediction_type'] = df.apply(classify_prediction, axis=1)

    # 各分類のカウント
    pred_counts = df['prediction_type'].value_counts()
    logger.info("\n【予測分類】")
    for pred_type, count in pred_counts.items():
        logger.info(f"  {pred_type}: {count} ({count/len(df)*100:.1f}%)")

    # FP と FN の特性比較
    fp = df[df['prediction_type'] == 'FP']
    fn = df[df['prediction_type'] == 'FN']
    tp = df[df['prediction_type'] == 'TP']

    logger.info(f"\nFalse Positive: {len(fp)}")
    logger.info(f"False Negative: {len(fn)}")
    logger.info(f"True Positive: {len(tp)}")

    # FPの特性
    if len(fp) > 0:
        logger.info("\n【False Positive の特性】")
        logger.info(f"  平均予測確率: {fp['predicted_prob'].mean():.3f}")
        logger.info(f"  平均プロジェクト数: {fp['project_count'].mean():.2f}")
        logger.info(f"  平均クロスプロジェクトスコア: {fp['cross_project_collaboration_score'].mean():.3f}")
        logger.info(f"  平均経験日数: {fp['experience_days'].mean():.1f}")
        logger.info(f"  平均最近承諾率: {fp['recent_acceptance_rate'].mean():.3f}")

    # FNの特性
    if len(fn) > 0:
        logger.info("\n【False Negative の特性】")
        logger.info(f"  平均予測確率: {fn['predicted_prob'].mean():.3f}")
        logger.info(f"  平均プロジェクト数: {fn['project_count'].mean():.2f}")
        logger.info(f"  平均クロスプロジェクトスコア: {fn['cross_project_collaboration_score'].mean():.3f}")
        logger.info(f"  平均経験日数: {fn['experience_days'].mean():.1f}")
        logger.info(f"  平均最近承諾率: {fn['recent_acceptance_rate'].mean():.3f}")

    # プロジェクトタイプ別の精度
    df['project_type'] = df['project_count'].apply(classify_project_type)
    accuracy_by_type = df.groupby('project_type').apply(
        lambda x: (x['pred_binary'] == x['true_label']).mean()
    ).sort_values(ascending=False)

    logger.info("\n【プロジェクトタイプ別予測精度】")
    for proj_type, acc in accuracy_by_type.items():
        count = len(df[df['project_type'] == proj_type])
        logger.info(f"  {proj_type:30s} {acc:.3f} (N={count})")

    # CSV保存
    error_analysis = pd.DataFrame({
        'prediction_type': pred_counts.index,
        'count': pred_counts.values,
        'percentage': (pred_counts.values / len(df)) * 100
    })
    error_analysis.to_csv(output_dir / 'prediction_type_distribution.csv', index=False)

    accuracy_df = pd.DataFrame({
        'project_type': accuracy_by_type.index,
        'accuracy': accuracy_by_type.values,
        'count': [len(df[df['project_type'] == pt]) for pt in accuracy_by_type.index]
    })
    accuracy_df.to_csv(output_dir / 'accuracy_by_project_type.csv', index=False)

    logger.info(f"\n分析結果を保存: {output_dir}")


def classify_project_type(count):
    """プロジェクト数からタイプを分類"""
    if count == 1:
        return 'Specialist (1 proj)'
    elif count <= 3:
        return 'Contributor (2-3 proj)'
    else:
        return 'Expert (4+ proj)'


def create_summary_report(
    importance_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    df: pd.DataFrame,
    output_dir: Path
):
    """
    サマリーレポートを作成
    """
    logger.info("=" * 80)
    logger.info("サマリーレポート作成")
    logger.info("=" * 80)

    # 予測結果
    df['pred_binary'] = (df['predicted_prob'] > 0.5).astype(int)
    accuracy = (df['pred_binary'] == df['true_label']).mean()

    # レポート作成
    report_lines = [
        "# 2x OSモデル詳細分析レポート",
        "",
        "## モデル性能",
        f"- **F1スコア**: 0.948",
        f"- **AUC-ROC**: 0.749",
        f"- **Precision**: 0.902",
        f"- **Recall**: 1.000",
        f"- **サンプル数**: {len(df)}",
        f"- **正例**: {df['true_label'].sum()} ({df['true_label'].mean()*100:.1f}%)",
        f"- **予測精度**: {accuracy:.3f}",
        "",
        "## Top 5 重要特徴量",
        ""
    ]

    for i, row in importance_df.head(5).iterrows():
        report_lines.append(f"{i+1}. **{row['feature_ja']}** ({row['feature']})")
        report_lines.append(f"   - Permutation Importance: {row['importance_mean']:.4f} ± {row['importance_std']:.4f}")
        report_lines.append(f"   - Random Forest Importance: {row['rf_importance']:.4f}")
        report_lines.append("")

    report_lines.extend([
        "## 予測成功 vs 失敗で差が大きい特徴量（Top 5）",
        ""
    ])

    for i, row in stats_df.nlargest(5, 'diff_pct').iterrows():
        report_lines.append(f"{i+1}. **{row['feature']}**")
        report_lines.append(f"   - 予測成功時: {row['correct_mean']:.3f} ± {row['correct_std']:.3f}")
        report_lines.append(f"   - 予測失敗時: {row['incorrect_mean']:.3f} ± {row['incorrect_std']:.3f}")
        report_lines.append(f"   - 差: {row['diff_pct']:+.1f}%")
        report_lines.append("")

    # プロジェクトタイプ別精度
    df['project_type'] = df['project_count'].apply(classify_project_type)
    accuracy_by_type = df.groupby('project_type').apply(
        lambda x: (x['pred_binary'] == x['true_label']).mean()
    ).sort_values(ascending=False)

    report_lines.extend([
        "## プロジェクトタイプ別予測精度",
        ""
    ])

    for proj_type, acc in accuracy_by_type.items():
        count = len(df[df['project_type'] == proj_type])
        report_lines.append(f"- **{proj_type}**: {acc:.3f} (N={count})")

    report_lines.append("")

    # ファイル保存
    report_path = output_dir / '2x_model_detailed_analysis_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    logger.info(f"\nレポートを保存: {report_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='2x OSモデルの詳細分析')
    parser.add_argument('--features', required=True, help='特徴量CSVパス')
    parser.add_argument('--output', required=True, help='出力ディレクトリ')

    args = parser.parse_args()

    # 出力ディレクトリ作成
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # データ読み込み
    logger.info(f"特徴量を読み込み: {args.features}")
    df = pd.read_csv(args.features)
    logger.info(f"レコード数: {len(df)}")

    # 分析実行
    importance_df = analyze_feature_importance(df, output_dir)
    stats_df = analyze_correct_vs_incorrect_predictions(df, output_dir)
    analyze_false_positive_false_negative(df, output_dir)
    create_summary_report(importance_df, stats_df, df, output_dir)

    logger.info("\n" + "=" * 80)
    logger.info("すべての分析が完了しました！")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
予測成功/失敗開発者の詳細分布分析

1. レビュー回数分布
2. メールドメイン（企業アカウント）分布
3. 各種特徴量の分布比較
"""

import logging
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_email_domain(email):
    """メールアドレスからドメインを抽出"""
    try:
        domain = email.split('@')[1]
        return domain
    except:
        return 'unknown'


def classify_domain_type(domain):
    """ドメインを企業/個人に分類"""
    # 主要企業ドメイン
    corporate_domains = {
        'redhat.com': 'Red Hat',
        'ibm.com': 'IBM',
        'mirantis.com': 'Mirantis',
        'canonical.com': 'Canonical',
        'suse.com': 'SUSE',
        'intel.com': 'Intel',
        'huawei.com': 'Huawei',
        'cisco.com': 'Cisco',
        'dell.com': 'Dell',
        'hp.com': 'HP',
        'oracle.com': 'Oracle',
        'vmware.com': 'VMware',
        'rackspace.com': 'Rackspace',
        'ovh.com': 'OVH',
        'nvidia.com': 'NVIDIA',
        'fujitsu.com': 'Fujitsu',
        'ericsson.com': 'Ericsson',
        'stackhpc.com': 'StackHPC',
        'cern.ch': 'CERN',
    }

    # フリーメールドメイン
    free_domains = {
        'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com',
        'protonmail.com', 'icloud.com', 'mail.com', 'aol.com'
    }

    if domain in corporate_domains:
        return corporate_domains[domain]
    elif domain in free_domains:
        return 'Personal (Gmail等)'
    elif domain.endswith('.edu') or domain.endswith('.ac.uk') or domain.endswith('.ac.jp'):
        return 'Academic'
    else:
        # 個人ドメインっぽいもの
        if any(x in domain for x in ['github', 'dev', 'personal', 'me.com']):
            return 'Personal'
        # それ以外は小規模企業として扱う
        return 'Other Company'


def analyze_distribution(df: pd.DataFrame, output_dir: Path):
    """
    予測成功/失敗の詳細分布分析
    """
    logger.info("=" * 80)
    logger.info("予測成功/失敗の詳細分布分析")
    logger.info("=" * 80)

    # 予測結果を分類
    df['pred_binary'] = (df['predicted_prob'] > 0.5).astype(int)
    df['is_correct'] = (df['pred_binary'] == df['true_label']).astype(int)

    correct = df[df['is_correct'] == 1]
    incorrect = df[df['is_correct'] == 0]

    logger.info(f"\n予測成功: {len(correct)} ({len(correct)/len(df)*100:.1f}%)")
    logger.info(f"予測失敗: {len(incorrect)} ({len(incorrect)/len(df)*100:.1f}%)")

    # メールドメイン抽出
    df['email_domain'] = df['reviewer_email'].apply(extract_email_domain)
    df['domain_type'] = df['email_domain'].apply(classify_domain_type)

    correct['email_domain'] = correct['reviewer_email'].apply(extract_email_domain)
    correct['domain_type'] = correct['email_domain'].apply(classify_domain_type)
    incorrect['email_domain'] = incorrect['reviewer_email'].apply(extract_email_domain)
    incorrect['domain_type'] = incorrect['email_domain'].apply(classify_domain_type)

    # 分析実施
    analyze_review_count_distribution(correct, incorrect, output_dir)
    analyze_domain_distribution(correct, incorrect, df, output_dir)
    analyze_feature_distributions(correct, incorrect, output_dir)
    create_developer_list(correct, incorrect, df, output_dir)


def analyze_review_count_distribution(correct: pd.DataFrame, incorrect: pd.DataFrame, output_dir: Path):
    """
    レビュー回数の分布分析
    """
    logger.info("\n" + "=" * 80)
    logger.info("レビュー回数分布分析")
    logger.info("=" * 80)

    # 統計量
    stats = []
    for label, data in [('Correct', correct), ('Incorrect', incorrect)]:
        stats.append({
            'prediction': label,
            'history_mean': data['history_request_count'].mean(),
            'history_median': data['history_request_count'].median(),
            'history_std': data['history_request_count'].std(),
            'eval_mean': data['eval_request_count'].mean(),
            'eval_median': data['eval_request_count'].median(),
            'eval_std': data['eval_request_count'].std(),
            'total_reviews_mean': data['total_reviews'].mean(),
            'total_reviews_median': data['total_reviews'].median(),
        })

    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(output_dir / 'review_count_stats.csv', index=False)

    logger.info("\n【レビュー回数統計】")
    logger.info(f"予測成功:")
    logger.info(f"  訓練期間: 平均 {stats[0]['history_mean']:.1f}件, 中央値 {stats[0]['history_median']:.1f}件")
    logger.info(f"  評価期間: 平均 {stats[0]['eval_mean']:.1f}件, 中央値 {stats[0]['eval_median']:.1f}件")
    logger.info(f"  総レビュー数: 平均 {stats[0]['total_reviews_mean']:.1f}件")
    logger.info(f"予測失敗:")
    logger.info(f"  訓練期間: 平均 {stats[1]['history_mean']:.1f}件, 中央値 {stats[1]['history_median']:.1f}件")
    logger.info(f"  評価期間: 平均 {stats[1]['eval_mean']:.1f}件, 中央値 {stats[1]['eval_median']:.1f}件")
    logger.info(f"  総レビュー数: 平均 {stats[1]['total_reviews_mean']:.1f}件")

    # 可視化
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (1) 訓練期間レビュー回数 - ヒストグラム
    axes[0, 0].hist(correct['history_request_count'], bins=30, alpha=0.6, label='Correct', color='green', edgecolor='black')
    axes[0, 0].hist(incorrect['history_request_count'], bins=30, alpha=0.6, label='Incorrect', color='red', edgecolor='black')
    axes[0, 0].set_xlabel('History Request Count', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Training Period Review Count Distribution', fontsize=13, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # (2) 評価期間レビュー回数 - ヒストグラム
    axes[0, 1].hist(correct['eval_request_count'], bins=30, alpha=0.6, label='Correct', color='green', edgecolor='black')
    axes[0, 1].hist(incorrect['eval_request_count'], bins=30, alpha=0.6, label='Incorrect', color='red', edgecolor='black')
    axes[0, 1].set_xlabel('Eval Request Count', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('Evaluation Period Review Count Distribution', fontsize=13, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # (3) 総レビュー数 - ヒストグラム
    axes[0, 2].hist(correct['total_reviews'], bins=30, alpha=0.6, label='Correct', color='green', edgecolor='black')
    axes[0, 2].hist(incorrect['total_reviews'], bins=30, alpha=0.6, label='Incorrect', color='red', edgecolor='black')
    axes[0, 2].set_xlabel('Total Reviews', fontsize=12)
    axes[0, 2].set_ylabel('Frequency', fontsize=12)
    axes[0, 2].set_title('Total Review Count Distribution', fontsize=13, fontweight='bold')
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.3)

    # (4) 訓練期間レビュー回数 - ボックスプロット
    data_to_plot = [correct['history_request_count'].dropna(), incorrect['history_request_count'].dropna()]
    bp1 = axes[1, 0].boxplot(data_to_plot, labels=['Correct', 'Incorrect'], patch_artist=True)
    bp1['boxes'][0].set_facecolor('lightgreen')
    bp1['boxes'][1].set_facecolor('lightcoral')
    axes[1, 0].set_ylabel('History Request Count', fontsize=12)
    axes[1, 0].set_title('Training Period Review Count (Boxplot)', fontsize=13, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)

    # (5) 評価期間レビュー回数 - ボックスプロット
    data_to_plot = [correct['eval_request_count'].dropna(), incorrect['eval_request_count'].dropna()]
    bp2 = axes[1, 1].boxplot(data_to_plot, labels=['Correct', 'Incorrect'], patch_artist=True)
    bp2['boxes'][0].set_facecolor('lightgreen')
    bp2['boxes'][1].set_facecolor('lightcoral')
    axes[1, 1].set_ylabel('Eval Request Count', fontsize=12)
    axes[1, 1].set_title('Evaluation Period Review Count (Boxplot)', fontsize=13, fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)

    # (6) 総レビュー数 - ボックスプロット
    data_to_plot = [correct['total_reviews'].dropna(), incorrect['total_reviews'].dropna()]
    bp3 = axes[1, 2].boxplot(data_to_plot, labels=['Correct', 'Incorrect'], patch_artist=True)
    bp3['boxes'][0].set_facecolor('lightgreen')
    bp3['boxes'][1].set_facecolor('lightcoral')
    axes[1, 2].set_ylabel('Total Reviews', fontsize=12)
    axes[1, 2].set_title('Total Review Count (Boxplot)', fontsize=13, fontweight='bold')
    axes[1, 2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'review_count_distribution.png', dpi=300, bbox_inches='tight')
    logger.info(f"可視化を保存: {output_dir / 'review_count_distribution.png'}")
    plt.close()


def analyze_domain_distribution(correct: pd.DataFrame, incorrect: pd.DataFrame, df: pd.DataFrame, output_dir: Path):
    """
    メールドメイン（企業アカウント）分布分析
    """
    logger.info("\n" + "=" * 80)
    logger.info("メールドメイン分布分析")
    logger.info("=" * 80)

    # ドメインタイプ別の統計
    domain_stats = []
    for domain_type in df['domain_type'].unique():
        total = len(df[df['domain_type'] == domain_type])
        correct_count = len(correct[correct['domain_type'] == domain_type])
        incorrect_count = len(incorrect[incorrect['domain_type'] == domain_type])
        accuracy = correct_count / total if total > 0 else 0

        domain_stats.append({
            'domain_type': domain_type,
            'total': total,
            'correct': correct_count,
            'incorrect': incorrect_count,
            'accuracy': accuracy
        })

    domain_stats_df = pd.DataFrame(domain_stats).sort_values('total', ascending=False)
    domain_stats_df.to_csv(output_dir / 'domain_type_stats.csv', index=False)

    logger.info("\n【ドメインタイプ別統計】")
    for _, row in domain_stats_df.iterrows():
        logger.info(f"  {row['domain_type']:30s} 総数:{row['total']:3d}, 成功:{row['correct']:3d}, 失敗:{row['incorrect']:3d}, 精度:{row['accuracy']:.3f}")

    # 主要ドメイン別の統計（Top 15）
    domain_counts = df['email_domain'].value_counts().head(15)
    domain_detail_stats = []

    for domain in domain_counts.index:
        total = len(df[df['email_domain'] == domain])
        correct_count = len(correct[correct['email_domain'] == domain])
        incorrect_count = len(incorrect[incorrect['email_domain'] == domain])
        accuracy = correct_count / total if total > 0 else 0
        domain_type = classify_domain_type(domain)

        domain_detail_stats.append({
            'domain': domain,
            'domain_type': domain_type,
            'total': total,
            'correct': correct_count,
            'incorrect': incorrect_count,
            'accuracy': accuracy
        })

    domain_detail_df = pd.DataFrame(domain_detail_stats)
    domain_detail_df.to_csv(output_dir / 'domain_detail_stats.csv', index=False)

    logger.info("\n【Top 15 ドメイン詳細】")
    for _, row in domain_detail_df.iterrows():
        logger.info(f"  {row['domain']:40s} ({row['domain_type']:20s}) 総数:{row['total']:3d}, 精度:{row['accuracy']:.3f}")

    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # (1) ドメインタイプ別の総数
    axes[0, 0].bar(range(len(domain_stats_df)), domain_stats_df['total'], color='steelblue', alpha=0.7)
    axes[0, 0].set_xticks(range(len(domain_stats_df)))
    axes[0, 0].set_xticklabels(domain_stats_df['domain_type'], rotation=45, ha='right')
    axes[0, 0].set_ylabel('Count', fontsize=12)
    axes[0, 0].set_title('Developer Count by Domain Type', fontsize=13, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)

    # (2) ドメインタイプ別の精度
    colors = ['green' if acc > 0.8 else 'orange' if acc > 0.6 else 'red' for acc in domain_stats_df['accuracy']]
    axes[0, 1].bar(range(len(domain_stats_df)), domain_stats_df['accuracy'], color=colors, alpha=0.7)
    axes[0, 1].set_xticks(range(len(domain_stats_df)))
    axes[0, 1].set_xticklabels(domain_stats_df['domain_type'], rotation=45, ha='right')
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].set_ylim(0, 1.0)
    axes[0, 1].axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='80%')
    axes[0, 1].set_title('Prediction Accuracy by Domain Type', fontsize=13, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)

    # (3) 予測成功/失敗のドメインタイプ分布（円グラフ）
    correct_domain_counts = correct['domain_type'].value_counts()
    axes[1, 0].pie(correct_domain_counts.values, labels=correct_domain_counts.index, autopct='%1.1f%%', startangle=90)
    axes[1, 0].set_title('Domain Type Distribution (Correct Predictions)', fontsize=13, fontweight='bold')

    incorrect_domain_counts = incorrect['domain_type'].value_counts()
    axes[1, 1].pie(incorrect_domain_counts.values, labels=incorrect_domain_counts.index, autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title('Domain Type Distribution (Incorrect Predictions)', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'domain_distribution.png', dpi=300, bbox_inches='tight')
    logger.info(f"可視化を保存: {output_dir / 'domain_distribution.png'}")
    plt.close()


def analyze_feature_distributions(correct: pd.DataFrame, incorrect: pd.DataFrame, output_dir: Path):
    """
    各種特徴量の分布比較
    """
    logger.info("\n" + "=" * 80)
    logger.info("特徴量分布比較")
    logger.info("=" * 80)

    features_to_plot = [
        'project_count',
        'recent_activity_frequency',
        'recent_acceptance_rate',
        'experience_days',
        'avg_activity_gap',
        'cross_project_collaboration_score'
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, feat in enumerate(features_to_plot):
        # ヒストグラム
        axes[i].hist(correct[feat].dropna(), bins=20, alpha=0.6, label='Correct', color='green', edgecolor='black', density=True)
        axes[i].hist(incorrect[feat].dropna(), bins=20, alpha=0.6, label='Incorrect', color='red', edgecolor='black', density=True)

        # 平均値のライン
        correct_mean = correct[feat].mean()
        incorrect_mean = incorrect[feat].mean()
        axes[i].axvline(correct_mean, color='darkgreen', linestyle='--', linewidth=2, label=f'Correct Mean: {correct_mean:.2f}')
        axes[i].axvline(incorrect_mean, color='darkred', linestyle='--', linewidth=2, label=f'Incorrect Mean: {incorrect_mean:.2f}')

        axes[i].set_xlabel(feat, fontsize=11)
        axes[i].set_ylabel('Density', fontsize=11)
        axes[i].set_title(f'{feat} Distribution', fontsize=12, fontweight='bold')
        axes[i].legend(fontsize=8)
        axes[i].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'feature_distributions.png', dpi=300, bbox_inches='tight')
    logger.info(f"可視化を保存: {output_dir / 'feature_distributions.png'}")
    plt.close()


def create_developer_list(correct: pd.DataFrame, incorrect: pd.DataFrame, df: pd.DataFrame, output_dir: Path):
    """
    開発者リストの作成（メールアドレス付き）
    """
    logger.info("\n" + "=" * 80)
    logger.info("開発者リスト作成")
    logger.info("=" * 80)

    # 予測成功開発者リスト
    correct_list = correct[[
        'reviewer_email', 'email_domain', 'domain_type',
        'predicted_prob', 'true_label',
        'history_request_count', 'eval_request_count',
        'total_reviews', 'project_count',
        'recent_activity_frequency', 'recent_acceptance_rate'
    ]].copy()
    correct_list['prediction_result'] = 'Correct'
    correct_list = correct_list.sort_values('project_count', ascending=False)
    correct_list.to_csv(output_dir / 'correct_developers.csv', index=False)

    logger.info(f"予測成功開発者リストを保存: {output_dir / 'correct_developers.csv'} ({len(correct_list)}名)")

    # 予測失敗開発者リスト
    incorrect_list = incorrect[[
        'reviewer_email', 'email_domain', 'domain_type',
        'predicted_prob', 'true_label',
        'history_request_count', 'eval_request_count',
        'total_reviews', 'project_count',
        'recent_activity_frequency', 'recent_acceptance_rate'
    ]].copy()
    incorrect_list['prediction_result'] = 'Incorrect'
    incorrect_list = incorrect_list.sort_values('project_count', ascending=False)
    incorrect_list.to_csv(output_dir / 'incorrect_developers.csv', index=False)

    logger.info(f"予測失敗開発者リストを保存: {output_dir / 'incorrect_developers.csv'} ({len(incorrect_list)}名)")

    # 統合リスト
    all_list = pd.concat([correct_list, incorrect_list], ignore_index=True)
    all_list = all_list.sort_values(['prediction_result', 'project_count'], ascending=[True, False])
    all_list.to_csv(output_dir / 'all_developers.csv', index=False)

    logger.info(f"統合開発者リストを保存: {output_dir / 'all_developers.csv'} ({len(all_list)}名)")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='予測成功/失敗の詳細分布分析')
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
    analyze_distribution(df, output_dir)

    logger.info("\n" + "=" * 80)
    logger.info("すべての分析が完了しました！")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

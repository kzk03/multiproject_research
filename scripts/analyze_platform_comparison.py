#!/usr/bin/env python3
"""
4プラットフォーム（OpenStack、Qt、Android、Chromium）の比較分析

1. データ統計の比較
2. 予測性能の比較
3. 特徴量分布の比較
4. プロジェクト間のばらつき分析
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class PlatformComparator:
    def __init__(self):
        self.platforms = {
            'OpenStack': {
                'data': 'data/openstack_50proj_2021_2024_feat.csv',
                'results': 'outputs/50projects_irl_improved/no_os'
            },
            'Qt': {
                'data': 'data/qt_50proj_2021_2024_feat.csv',
                'results': 'outputs/qt_50projects_irl'
            },
            'Android': {
                'data': 'data/android_50proj_2021_2024_feat.csv',
                'results': 'outputs/android_50projects_irl'
            },
            'Chromium': {
                'data': 'data/chromium_50proj_2021_2024_feat.csv',
                'results': 'outputs/chromium_50projects_irl'
            }
        }

        self.stats = {}
        self.performance = {}

    def load_all_data(self):
        """全プラットフォームのデータを読み込み"""
        print("=" * 80)
        print("データ読み込み中...")
        print("=" * 80)

        for platform, paths in self.platforms.items():
            data_path = Path(paths['data'])
            if data_path.exists():
                df = pd.read_csv(data_path)
                self.stats[platform] = {
                    'df': df,
                    'path': data_path
                }
                print(f"✓ {platform}: {len(df):,} レビューリクエスト")
            else:
                print(f"✗ {platform}: データが見つかりません ({data_path})")

    def compute_basic_statistics(self) -> pd.DataFrame:
        """基本統計量を計算"""
        print("\n" + "=" * 80)
        print("基本統計量の計算")
        print("=" * 80)

        stats_list = []

        for platform, data in self.stats.items():
            df = data['df']

            # 基本統計
            total_reviews = len(df)
            positive_rate = (df['label'] == 1).mean()
            negative_rate = (df['label'] == 0).mean()

            # プロジェクト数
            n_projects = df['project'].nunique() if 'project' in df.columns else 1

            # レビュアー数
            n_reviewers = df['reviewer_email'].nunique()

            # クロスプロジェクト活動
            cross_project_rate = df['is_cross_project'].mean() if 'is_cross_project' in df.columns else np.nan
            avg_project_count = df['reviewer_project_count'].mean() if 'reviewer_project_count' in df.columns else np.nan

            # 活動量の統計
            avg_past_reviews_30d = df['past_reviews_30d'].mean() if 'past_reviews_30d' in df.columns else np.nan
            avg_past_reviews_90d = df['past_reviews_90d'].mean() if 'past_reviews_90d' in df.columns else np.nan

            # パス類似度の統計
            avg_path_jaccard = df['path_jaccard_global'].mean() if 'path_jaccard_global' in df.columns else np.nan

            stats_list.append({
                'Platform': platform,
                'Total Reviews': total_reviews,
                'Positive Rate': positive_rate,
                'Negative Rate': negative_rate,
                'N Projects': n_projects,
                'N Reviewers': n_reviewers,
                'Cross-Project Rate': cross_project_rate,
                'Avg Project Count': avg_project_count,
                'Avg Past Reviews (30d)': avg_past_reviews_30d,
                'Avg Past Reviews (90d)': avg_past_reviews_90d,
                'Avg Path Jaccard': avg_path_jaccard,
            })

        stats_df = pd.DataFrame(stats_list)
        stats_df = stats_df.set_index('Platform')

        print("\n基本統計量:")
        print(stats_df.to_string())

        return stats_df

    def load_performance_metrics(self):
        """性能メトリクスを読み込み"""
        print("\n" + "=" * 80)
        print("性能メトリクスの読み込み")
        print("=" * 80)

        for platform, paths in self.platforms.items():
            results_dir = Path(paths['results'])

            metrics = {}
            for metric_name in ['f1_score', 'AUC_ROC', 'AUC_PR', 'PRECISION', 'RECALL']:
                metric_file = results_dir / f'matrix_{metric_name}.csv'
                if metric_file.exists():
                    df = pd.read_csv(metric_file, index_col=0)
                    metrics[metric_name] = df

                    # 対角線の平均（同期間評価）
                    diagonal_values = []
                    for i in range(min(df.shape)):
                        val = df.iloc[i, i]
                        if not pd.isna(val):
                            diagonal_values.append(val)

                    # 全体の平均（NaNを除く）
                    all_values = df.values.flatten()
                    all_values = all_values[~np.isnan(all_values)]

                    print(f"{platform} - {metric_name}:")
                    print(f"  対角平均: {np.mean(diagonal_values):.4f}" if diagonal_values else "  対角平均: N/A")
                    print(f"  全体平均: {np.mean(all_values):.4f}" if len(all_values) > 0 else "  全体平均: N/A")

            self.performance[platform] = metrics

    def create_comparison_matrices(self, output_dir: Path):
        """比較マトリクスを作成"""
        print("\n" + "=" * 80)
        print("比較マトリクスの作成")
        print("=" * 80)

        output_dir.mkdir(parents=True, exist_ok=True)

        # F1スコアの比較
        f1_comparison = {}
        for platform, metrics in self.performance.items():
            if 'f1_score' in metrics:
                df = metrics['f1_score']
                # 対角線の値を取得
                diagonal = []
                for i in range(min(df.shape)):
                    val = df.iloc[i, i]
                    if not pd.isna(val):
                        diagonal.append(val)

                # 全体の平均
                all_vals = df.values.flatten()
                all_vals = all_vals[~np.isnan(all_vals)]

                f1_comparison[platform] = {
                    'diagonal_mean': np.mean(diagonal) if diagonal else np.nan,
                    'overall_mean': np.mean(all_vals) if len(all_vals) > 0 else np.nan,
                    'diagonal_std': np.std(diagonal) if diagonal else np.nan,
                    'overall_std': np.std(all_vals) if len(all_vals) > 0 else np.nan,
                }

        comparison_df = pd.DataFrame(f1_comparison).T
        comparison_df.columns = ['F1 (Diagonal Mean)', 'F1 (Overall Mean)', 'F1 (Diagonal Std)', 'F1 (Overall Std)']

        print("\nF1スコア比較:")
        print(comparison_df.to_string())

        comparison_df.to_csv(output_dir / 'f1_comparison.csv')

        return comparison_df

    def analyze_feature_distributions(self, output_dir: Path):
        """特徴量分布を分析"""
        print("\n" + "=" * 80)
        print("特徴量分布の分析")
        print("=" * 80)

        # 共通の特徴量
        common_features = [
            'past_reviews_30d', 'past_reviews_90d', 'past_reviews_180d',
            'response_rate_30d', 'response_rate_90d',
            'path_jaccard_global', 'is_cross_project', 'reviewer_project_count'
        ]

        feature_stats = []

        for platform, data in self.stats.items():
            df = data['df']
            platform_stats = {'Platform': platform}

            for feature in common_features:
                if feature in df.columns:
                    platform_stats[f'{feature}_mean'] = df[feature].mean()
                    platform_stats[f'{feature}_std'] = df[feature].std()
                    platform_stats[f'{feature}_median'] = df[feature].median()
                else:
                    platform_stats[f'{feature}_mean'] = np.nan
                    platform_stats[f'{feature}_std'] = np.nan
                    platform_stats[f'{feature}_median'] = np.nan

            feature_stats.append(platform_stats)

        feature_df = pd.DataFrame(feature_stats)
        feature_df = feature_df.set_index('Platform')

        print("\n特徴量統計:")
        print(feature_df.to_string())

        feature_df.to_csv(output_dir / 'feature_statistics.csv')

        return feature_df

    def create_summary_report(self, output_dir: Path):
        """サマリーレポートを作成"""
        print("\n" + "=" * 80)
        print("サマリーレポートの作成")
        print("=" * 80)

        report_path = output_dir / 'platform_comparison_report.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("4プラットフォーム比較分析レポート\n")
            f.write("=" * 80 + "\n\n")

            # 基本統計
            f.write("## 1. 基本統計量\n\n")
            stats_df = self.compute_basic_statistics()
            f.write(stats_df.to_string() + "\n\n")

            # 性能比較
            f.write("## 2. 予測性能比較\n\n")
            for platform, metrics in self.performance.items():
                f.write(f"### {platform}\n")
                if 'f1_score' in metrics:
                    df = metrics['f1_score']
                    all_vals = df.values.flatten()
                    all_vals = all_vals[~np.isnan(all_vals)]
                    f.write(f"  F1スコア平均: {np.mean(all_vals):.4f}\n")
                    f.write(f"  F1スコア標準偏差: {np.std(all_vals):.4f}\n")
                f.write("\n")

            # 主な発見
            f.write("## 3. 主な発見\n\n")

            # データサイズ
            sizes = [(platform, data['df'].shape[0]) for platform, data in self.stats.items()]
            sizes.sort(key=lambda x: x[1], reverse=True)
            f.write("### データサイズ順:\n")
            for platform, size in sizes:
                f.write(f"  {platform}: {size:,} レビューリクエスト\n")
            f.write("\n")

            # クロスプロジェクト活動率
            f.write("### クロスプロジェクト活動率:\n")
            for platform, data in self.stats.items():
                df = data['df']
                if 'is_cross_project' in df.columns:
                    rate = df['is_cross_project'].mean()
                    f.write(f"  {platform}: {rate:.1%}\n")
            f.write("\n")

            # F1スコア順
            f1_scores = []
            for platform, metrics in self.performance.items():
                if 'f1_score' in metrics:
                    df = metrics['f1_score']
                    all_vals = df.values.flatten()
                    all_vals = all_vals[~np.isnan(all_vals)]
                    if len(all_vals) > 0:
                        f1_scores.append((platform, np.mean(all_vals)))

            f1_scores.sort(key=lambda x: x[1], reverse=True)
            f.write("### F1スコア順:\n")
            for platform, score in f1_scores:
                f.write(f"  {platform}: {score:.4f}\n")

        print(f"\nレポート保存: {report_path}")

        return report_path


def main():
    print("=" * 80)
    print("4プラットフォーム比較分析")
    print("OpenStack / Qt / Android / Chromium")
    print("=" * 80)

    # 出力ディレクトリ
    output_dir = Path('outputs/platform_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 比較分析実行
    comparator = PlatformComparator()
    comparator.load_all_data()

    # 基本統計
    stats_df = comparator.compute_basic_statistics()
    stats_df.to_csv(output_dir / 'basic_statistics.csv')

    # 性能メトリクス
    comparator.load_performance_metrics()

    # 比較マトリクス
    comparison_df = comparator.create_comparison_matrices(output_dir)

    # 特徴量分析
    feature_df = comparator.analyze_feature_distributions(output_dir)

    # サマリーレポート
    report_path = comparator.create_summary_report(output_dir)

    print("\n" + "=" * 80)
    print("分析完了！")
    print("=" * 80)
    print(f"\n結果は {output_dir} に保存されました")
    print("\n生成されたファイル:")
    print(f"  - basic_statistics.csv")
    print(f"  - f1_comparison.csv")
    print(f"  - feature_statistics.csv")
    print(f"  - platform_comparison_report.txt")


if __name__ == '__main__':
    main()

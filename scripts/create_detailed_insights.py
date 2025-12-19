#!/usr/bin/env python3
"""
詳細なインサイト分析

1. プロジェクト間のばらつき分析
2. データ特性と予測性能の相関
3. 各プラットフォームの特徴的なパターン
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List

def analyze_project_level_variance():
    """プロジェクトレベルのばらつき分析"""
    print("=" * 80)
    print("プロジェクトレベルのばらつき分析")
    print("=" * 80)

    platforms = {
        'OpenStack': 'data/openstack_50proj_2021_2024_feat.csv',
        'Qt': 'data/qt_50proj_2021_2024_feat.csv',
        'Android': 'data/android_50proj_2021_2024_feat.csv',
        'Chromium': 'data/chromium_50proj_2021_2024_feat.csv',
    }

    results = []

    for platform_name, data_path in platforms.items():
        df = pd.read_csv(data_path)

        # プロジェクトごとの統計
        project_stats = df.groupby('project').agg({
            'label': ['count', 'mean'],
            'is_cross_project': 'mean',
            'reviewer_project_count': 'mean'
        }).reset_index()

        project_stats.columns = ['project', 'n_reviews', 'positive_rate', 'cross_project_rate', 'avg_project_count']

        print(f"\n{platform_name}:")
        print(f"  プロジェクト数: {len(project_stats)}")
        print(f"  レビュー数の平均: {project_stats['n_reviews'].mean():.0f}")
        print(f"  レビュー数の標準偏差: {project_stats['n_reviews'].std():.0f}")
        print(f"  レビュー数の範囲: {project_stats['n_reviews'].min():.0f} - {project_stats['n_reviews'].max():.0f}")
        print(f"  承諾率の平均: {project_stats['positive_rate'].mean():.3f}")
        print(f"  承諾率の標準偏差: {project_stats['positive_rate'].std():.3f}")

        # 最大・最小のプロジェクト
        top5 = project_stats.nlargest(5, 'n_reviews')
        print(f"\n  レビュー数TOP5:")
        for _, row in top5.iterrows():
            print(f"    {row['project']}: {row['n_reviews']:.0f}件 (承諾率 {row['positive_rate']:.1%})")

        results.append({
            'Platform': platform_name,
            'N_Projects': len(project_stats),
            'Avg_Reviews_Per_Project': project_stats['n_reviews'].mean(),
            'Std_Reviews_Per_Project': project_stats['n_reviews'].std(),
            'Min_Reviews': project_stats['n_reviews'].min(),
            'Max_Reviews': project_stats['n_reviews'].max(),
            'Avg_Positive_Rate': project_stats['positive_rate'].mean(),
            'Std_Positive_Rate': project_stats['positive_rate'].std(),
        })

    variance_df = pd.DataFrame(results)
    variance_df.to_csv('outputs/platform_comparison/project_level_variance.csv', index=False)

    print("\n保存: outputs/platform_comparison/project_level_variance.csv")

    return variance_df


def analyze_correlation_patterns():
    """データ特性と性能の相関分析"""
    print("\n" + "=" * 80)
    print("データ特性と予測性能の相関分析")
    print("=" * 80)

    # 基本統計とF1スコアの関係
    stats = {
        'OpenStack': {'reviews': 119010, 'cross_project_rate': 0.856, 'f1': 0.9057, 'n_projects': 48},
        'Qt': {'reviews': 3121, 'cross_project_rate': 0.739, 'f1': 0.8161, 'n_projects': 39},
        'Android': {'reviews': 7155, 'cross_project_rate': 0.767, 'f1': 0.9434, 'n_projects': 26},
        'Chromium': {'reviews': 236, 'cross_project_rate': 0.047, 'f1': 0.8556, 'n_projects': 10},
    }

    correlation_df = pd.DataFrame(stats).T
    correlation_df = correlation_df.reset_index()
    correlation_df.columns = ['Platform', 'Total_Reviews', 'Cross_Project_Rate', 'F1_Score', 'N_Projects']

    print("\n相関分析:")
    print(correlation_df.to_string(index=False))

    # 相関係数
    print("\n相関係数:")
    print(f"  データサイズ vs F1スコア: {correlation_df['Total_Reviews'].corr(correlation_df['F1_Score']):.3f}")
    print(f"  クロスプロジェクト率 vs F1スコア: {correlation_df['Cross_Project_Rate'].corr(correlation_df['F1_Score']):.3f}")
    print(f"  プロジェクト数 vs F1スコア: {correlation_df['N_Projects'].corr(correlation_df['F1_Score']):.3f}")

    correlation_df.to_csv('outputs/platform_comparison/correlation_analysis.csv', index=False)
    print("\n保存: outputs/platform_comparison/correlation_analysis.csv")

    return correlation_df


def create_insights_summary():
    """インサイトサマリーを作成"""
    print("\n" + "=" * 80)
    print("主要インサイトのまとめ")
    print("=" * 80)

    insights = []

    # Insight 1: データサイズの影響
    insights.append({
        'category': 'データサイズの影響',
        'insight': 'OpenStackは最大のデータセット（119K）で高いF1スコア（0.906）を達成。'
                  'しかし、Androidは小規模（7K）でも最高のF1スコア（0.943）を記録。'
                  'データサイズが大きいほど良いとは限らない。',
        'implication': 'データの質とプロジェクトの性質が性能に大きく影響'
    })

    # Insight 2: クロスプロジェクト活動の影響
    insights.append({
        'category': 'クロスプロジェクト活動',
        'insight': 'OpenStack（85.6%）、Android（76.7%）、Qt（73.9%）は高いクロスプロジェクト活動率。'
                  'Chromiumは極端に低い（4.7%）が、F1スコアは0.856と中程度。',
        'implication': 'クロスプロジェクト活動が多いほど、レビュアーの継続性予測が容易になる可能性'
    })

    # Insight 3: 予測性能のばらつき
    insights.append({
        'category': '予測性能の安定性',
        'insight': 'Chromiumは最も高い標準偏差（0.151）で不安定。'
                  'OpenStackは最も安定（0.029）。'
                  'これはデータサイズと相関している可能性。',
        'implication': '小規模データセットでは時期による性能のばらつきが大きい'
    })

    # Insight 4: プラットフォーム特性
    insights.append({
        'category': 'プラットフォーム特性',
        'insight': 'Android: 最高のF1スコア、高いクロスプロジェクト活動、適度なデータサイズ'
                  '\nOpenStack: 大規模データ、高いクロスプロジェクト活動、安定した性能'
                  '\nQt: 中規模データ、やや低い性能、高いクロスプロジェクト活動'
                  '\nChromium: 小規模データ、低いクロスプロジェクト活動、不安定な性能',
        'implication': '各プラットフォームのコミュニティ構造が予測性能に影響'
    })

    # Insight 5: 承諾率の違い
    insights.append({
        'category': '承諾率の違い',
        'insight': 'Chromium（71.6%）とAndroid（67.1%）は高い承諾率。'
                  'Qt（43.8%）は最も低い承諾率。'
                  'OpenStackは中程度（61.7%）。',
        'implication': 'プラットフォームごとのレビュー文化の違いが顕著'
    })

    insights_df = pd.DataFrame(insights)

    # レポート作成
    report_path = Path('outputs/platform_comparison/key_insights.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("主要インサイト\n")
        f.write("=" * 80 + "\n\n")

        for i, insight in enumerate(insights, 1):
            f.write(f"## インサイト {i}: {insight['category']}\n\n")
            f.write(f"**発見:**\n{insight['insight']}\n\n")
            f.write(f"**示唆:**\n{insight['implication']}\n\n")
            f.write("-" * 80 + "\n\n")

        # 総合的な結論
        f.write("## 総合的な結論\n\n")
        f.write("1. **Androidが最も優れた予測性能を示す**\n")
        f.write("   - F1スコア: 0.943（最高）\n")
        f.write("   - 適度なデータサイズと高いクロスプロジェクト活動のバランスが良い\n\n")

        f.write("2. **OpenStackは大規模データで安定した性能**\n")
        f.write("   - 最大のデータセットで標準偏差が最小\n")
        f.write("   - 非常に高いクロスプロジェクト活動率（85.6%）\n\n")

        f.write("3. **Chromiumは小規模データの課題を示す**\n")
        f.write("   - わずか236レビューで高いばらつき\n")
        f.write("   - 低いクロスプロジェクト活動率（4.7%）が特徴的\n\n")

        f.write("4. **Qtは中間的な性能**\n")
        f.write("   - 低い承諾率（43.8%）がチャレンジング\n")
        f.write("   - 中規模データセットで中程度の性能\n\n")

        f.write("## 推奨事項\n\n")
        f.write("1. **データ収集の拡大**: Chromiumのようにデータが少ないプラットフォームでは、\n")
        f.write("   より長期間のデータ収集または追加プロジェクトの検討が必要\n\n")

        f.write("2. **クロスプロジェクト特徴の活用**: 高いクロスプロジェクト活動率を持つ\n")
        f.write("   プラットフォームでは、この特徴を積極的に活用すべき\n\n")

        f.write("3. **プラットフォーム固有の調整**: 各プラットフォームの特性に応じた\n")
        f.write("   モデルパラメータの調整が性能向上の鍵\n\n")

    print(f"\nインサイトレポート保存: {report_path}")

    return insights_df


def main():
    print("=" * 80)
    print("詳細インサイト分析")
    print("=" * 80)

    # プロジェクトレベルのばらつき分析
    variance_df = analyze_project_level_variance()

    # 相関分析
    correlation_df = analyze_correlation_patterns()

    # インサイトサマリー
    insights_df = create_insights_summary()

    print("\n" + "=" * 80)
    print("分析完了！")
    print("=" * 80)
    print("\n生成されたファイル:")
    print("  - outputs/platform_comparison/project_level_variance.csv")
    print("  - outputs/platform_comparison/correlation_analysis.csv")
    print("  - outputs/platform_comparison/key_insights.txt")


if __name__ == '__main__':
    main()

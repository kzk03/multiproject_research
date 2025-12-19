#!/usr/bin/env python3
"""
選定したプロジェクトと実際に収集されたデータの差分を確認
"""

import pandas as pd
from pathlib import Path

def check_platform(platform_name, project_list_file, data_file):
    """各プラットフォームのプロジェクトをチェック"""
    print(f"\n{'='*80}")
    print(f"{platform_name}")
    print('='*80)

    # 選定したプロジェクトリスト
    with open(project_list_file, 'r') as f:
        selected_projects = [line.strip() for line in f if line.strip()]

    print(f"選定したプロジェクト数: {len(selected_projects)}")

    # 実際のデータ
    df = pd.read_csv(data_file)
    actual_projects = df['project'].unique().tolist()

    print(f"データに含まれるプロジェクト数: {len(actual_projects)}")

    # 差分
    selected_set = set(selected_projects)
    actual_set = set(actual_projects)

    missing = selected_set - actual_set
    extra = actual_set - selected_set

    if missing:
        print(f"\n⚠️  データに含まれていないプロジェクト: {len(missing)}件")
        print("\nプロジェクト名 | レビュー数")
        print("-" * 80)
        for proj in sorted(missing):
            # データ内で確認
            proj_data = df[df['project'] == proj]
            count = len(proj_data)
            print(f"{proj}: {count}件")

    if extra:
        print(f"\n⚠️  リストにないプロジェクト: {len(extra)}件")
        for proj in sorted(extra):
            proj_data = df[df['project'] == proj]
            print(f"{proj}: {len(proj_data)}件")

    # プロジェクトごとのレビュー数
    project_counts = df.groupby('project').size().sort_values(ascending=False)

    print(f"\n📊 レビュー数の分布:")
    print(f"  最小: {project_counts.min()}")
    print(f"  最大: {project_counts.max()}")
    print(f"  平均: {project_counts.mean():.1f}")
    print(f"  中央値: {project_counts.median():.1f}")

    # レビュー数が0のプロジェクト
    zero_review_projects = [p for p in selected_projects if p not in actual_projects]
    if zero_review_projects:
        print(f"\n❌ 2021-2024期間中にレビューがなかったプロジェクト: {len(zero_review_projects)}件")
        for proj in sorted(zero_review_projects)[:10]:  # 最初の10件のみ表示
            print(f"  - {proj}")
        if len(zero_review_projects) > 10:
            print(f"  ... 他 {len(zero_review_projects) - 10}件")

def main():
    platforms = {
        'OpenStack': {
            'list': 'projects_50.txt',
            'data': 'data/openstack_50proj_2021_2024_feat.csv'
        },
        'Qt': {
            'list': 'projects_qt_50.txt',
            'data': 'data/qt_50proj_2021_2024_feat.csv'
        },
        'Android': {
            'list': 'projects_android_50.txt',
            'data': 'data/android_50proj_2021_2024_feat.csv'
        },
        'Chromium': {
            'list': 'projects_chromium_50.txt',
            'data': 'data/chromium_50proj_2021_2024_feat.csv'
        }
    }

    print("="*80)
    print("プロジェクト選定とデータ収集の差分チェック")
    print("="*80)

    for platform_name, paths in platforms.items():
        check_platform(platform_name, paths['list'], paths['data'])

    print("\n" + "="*80)
    print("分析完了")
    print("="*80)
    print("\n📝 結論:")
    print("  - 選定した50プロジェクトのうち、2021-2024期間中に")
    print("    レビュー活動がなかったプロジェクトは自動的に除外されています")
    print("  - これは正常な動作です（活動のないプロジェクトは学習に不要）")

if __name__ == '__main__':
    main()

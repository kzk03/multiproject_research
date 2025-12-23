"""
プロジェクト間のIRL統計を比較表示するスクリプト
"""
import pandas as pd
from pathlib import Path

def main():
    stats_dir = Path("/Users/kazuki-h/research/multiproject_research/results/irl_project_stats")

    # サマリーを読み込み
    summary = pd.read_csv(stats_dir / "project_summary.csv")

    print("\n" + "="*80)
    print("プロジェクト別統計サマリー (IRL)")
    print("="*80)

    print("\n【1. 開発者単位の継続率】")
    print("-"*80)
    continuation_stats = summary[['project', 'total_developers', 'continuing_developers',
                                  'dropout_developers', 'developer_continuation_rate']].copy()
    continuation_stats['developer_continuation_rate'] = (continuation_stats['developer_continuation_rate'] * 100).round(2)
    continuation_stats = continuation_stats.rename(columns={
        'project': 'プロジェクト',
        'total_developers': '総開発者数',
        'continuing_developers': '継続開発者数',
        'dropout_developers': 'ドロップアウト数',
        'developer_continuation_rate': '継続率 (%)'
    })
    continuation_stats = continuation_stats.sort_values('継続率 (%)', ascending=False)
    print(continuation_stats.to_string(index=False))

    print("\n【2. レビュー単位の承諾率（過去）】")
    print("-"*80)
    history_acc = summary[['project', 'avg_history_acceptance_rate']].copy()
    history_acc['avg_history_acceptance_rate'] = (history_acc['avg_history_acceptance_rate'] * 100).round(2)
    history_acc = history_acc.rename(columns={
        'project': 'プロジェクト',
        'avg_history_acceptance_rate': '平均承諾率 (%)'
    })
    history_acc = history_acc.sort_values('平均承諾率 (%)', ascending=False)
    print(history_acc.to_string(index=False))

    print("\n【3. レビュー単位の承諾率（評価期間）】")
    print("-"*80)
    eval_acc = summary[['project', 'avg_eval_acceptance_rate', 'overall_eval_acceptance_rate',
                        'total_eval_requests', 'total_eval_accepted']].copy()
    eval_acc['avg_eval_acceptance_rate'] = (eval_acc['avg_eval_acceptance_rate'] * 100).round(2)
    eval_acc['overall_eval_acceptance_rate'] = (eval_acc['overall_eval_acceptance_rate'] * 100).round(2)
    eval_acc = eval_acc.rename(columns={
        'project': 'プロジェクト',
        'avg_eval_acceptance_rate': '平均承諾率 (%)',
        'overall_eval_acceptance_rate': '全体承諾率 (%)',
        'total_eval_requests': '総レビュー数',
        'total_eval_accepted': '承諾レビュー数'
    })
    eval_acc = eval_acc.sort_values('全体承諾率 (%)', ascending=False)
    print(eval_acc.to_string(index=False))

    print("\n【4. プロジェクト規模の比較】")
    print("-"*80)
    scale = summary[['project', 'total_developers', 'total_eval_requests']].copy()
    scale['avg_reviews_per_dev'] = (scale['total_eval_requests'] / scale['total_developers']).round(2)
    scale = scale.rename(columns={
        'project': 'プロジェクト',
        'total_developers': '総開発者数',
        'total_eval_requests': '総レビュー数',
        'avg_reviews_per_dev': '開発者あたりレビュー数'
    })
    scale = scale.sort_values('総レビュー数', ascending=False)
    print(scale.to_string(index=False))

    print("\n" + "="*80)
    print("各プロジェクトの詳細統計ファイル:")
    print("-"*80)
    for project in ['qt', 'android', 'chromium', 'openstack']:
        print(f"  {stats_dir / f'{project}_stats.csv'}")
    print(f"\n全プロジェクト統計: {stats_dir / 'all_projects_stats.csv'}")
    print("="*80)

if __name__ == "__main__":
    main()

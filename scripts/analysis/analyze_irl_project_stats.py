"""
各プロジェクトのIRL結果から、レビュー単位の承諾率と開発者単位の継続率を分析するスクリプト
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
import glob

def analyze_project(project_path, project_name):
    """プロジェクトごとの統計を分析"""
    results = []

    # すべてのpredictions.csvファイルを探す
    csv_files = glob.glob(str(project_path / "**" / "predictions.csv"), recursive=True)

    for csv_file in csv_files:
        # パスから学習期間と評価期間を抽出
        path_parts = Path(csv_file).parts
        train_period = None
        eval_period = None

        for i, part in enumerate(path_parts):
            if part.startswith('train_'):
                train_period = part.replace('train_', '')
            if part.startswith('eval_'):
                eval_period = part.replace('eval_', '')

        if not train_period or not eval_period:
            continue

        # データを読み込み
        df = pd.read_csv(csv_file)

        if len(df) == 0:
            continue

        # レビュー単位の承諾率（評価期間）
        df['eval_acceptance_rate'] = df.apply(
            lambda row: row['eval_accepted_count'] / row['eval_request_count']
            if row['eval_request_count'] > 0 else np.nan,
            axis=1
        )

        # 統計を計算
        stats = {
            'project': project_name,
            'train_period': train_period,
            'eval_period': eval_period,

            # 開発者数
            'total_developers': len(df),
            'continuing_developers': df['true_label'].sum(),
            'dropout_developers': len(df) - df['true_label'].sum(),

            # 開発者単位の継続率
            'developer_continuation_rate': df['true_label'].mean(),

            # レビュー単位の承諾率（過去）
            'avg_history_acceptance_rate': df['history_acceptance_rate'].mean(),
            'median_history_acceptance_rate': df['history_acceptance_rate'].median(),
            'std_history_acceptance_rate': df['history_acceptance_rate'].std(),

            # レビュー単位の承諾率（評価期間）
            'avg_eval_acceptance_rate': df['eval_acceptance_rate'].mean(),
            'median_eval_acceptance_rate': df['eval_acceptance_rate'].median(),
            'std_eval_acceptance_rate': df['eval_acceptance_rate'].std(),

            # レビュー総数
            'total_eval_requests': df['eval_request_count'].sum(),
            'total_eval_accepted': df['eval_accepted_count'].sum(),
            'total_eval_rejected': df['eval_rejected_count'].sum(),

            # 全体の承諾率
            'overall_eval_acceptance_rate': df['eval_accepted_count'].sum() / df['eval_request_count'].sum()
            if df['eval_request_count'].sum() > 0 else np.nan,
        }

        results.append(stats)

    return pd.DataFrame(results)


def main():
    base_path = Path("/Users/kazuki-h/research/multiproject_research/outputs")

    projects = {
        'Qt': base_path / "qt_50projects_irl",
        'Android': base_path / "android_50projects_irl",
        'Chromium': base_path / "chromium_50projects_irl",
        'OpenStack': base_path / "opnestack_50projects_irl_timeseries" / "2x_os",
    }

    all_results = []

    for project_name, project_path in projects.items():
        print(f"\n{'='*60}")
        print(f"分析中: {project_name}")
        print(f"{'='*60}")

        if not project_path.exists():
            print(f"警告: {project_path} が見つかりません")
            continue

        project_results = analyze_project(project_path, project_name)

        if len(project_results) > 0:
            all_results.append(project_results)

            print(f"\n{project_name} の結果:")
            print(project_results.to_string(index=False))
        else:
            print(f"{project_name}: データが見つかりませんでした")

    # 全プロジェクトの結果を結合
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)

        # 結果を保存
        output_dir = Path("/Users/kazuki-h/research/multiproject_research/results/irl_project_stats")
        output_dir.mkdir(parents=True, exist_ok=True)

        combined_df.to_csv(output_dir / "all_projects_stats.csv", index=False)
        print(f"\n\n{'='*60}")
        print("結果を保存しました:")
        print(f"  {output_dir / 'all_projects_stats.csv'}")

        # プロジェクトごとのサマリー
        print(f"\n{'='*60}")
        print("プロジェクト別サマリー")
        print(f"{'='*60}")

        summary = combined_df.groupby('project').agg({
            'total_developers': 'sum',
            'continuing_developers': 'sum',
            'dropout_developers': 'sum',
            'developer_continuation_rate': 'mean',
            'avg_history_acceptance_rate': 'mean',
            'avg_eval_acceptance_rate': 'mean',
            'overall_eval_acceptance_rate': 'mean',
            'total_eval_requests': 'sum',
            'total_eval_accepted': 'sum',
        }).round(4)

        print(summary)
        summary.to_csv(output_dir / "project_summary.csv")
        print(f"\nサマリーを保存しました: {output_dir / 'project_summary.csv'}")

        # 各プロジェクトの詳細を個別ファイルに保存
        for project_name in combined_df['project'].unique():
            project_df = combined_df[combined_df['project'] == project_name]
            project_df.to_csv(output_dir / f"{project_name.lower()}_stats.csv", index=False)
            print(f"  {output_dir / f'{project_name.lower()}_stats.csv'}")


if __name__ == "__main__":
    main()

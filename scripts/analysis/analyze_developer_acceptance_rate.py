"""
開発者ごとのレビュー承諾率を分析するスクリプト
レビュー承諾 = アサインされた時にレビューを引き受けるかどうか
"""
import pandas as pd
import numpy as np
from pathlib import Path
import glob

def analyze_developer_acceptance(project_path, project_name):
    """プロジェクトごとの開発者別承諾率を分析"""
    all_developers = []

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

        # 各開発者の承諾率を計算
        df['acceptance_rate'] = df.apply(
            lambda row: row['eval_accepted_count'] / row['eval_request_count']
            if row['eval_request_count'] > 0 else np.nan,
            axis=1
        )

        # プロジェクト情報を追加
        df['project'] = project_name
        df['train_period'] = train_period
        df['eval_period'] = eval_period

        all_developers.append(df)

    if all_developers:
        return pd.concat(all_developers, ignore_index=True)
    else:
        return pd.DataFrame()


def main():
    base_path = Path("/Users/kazuki-h/research/multiproject_research/outputs/multiprojects")

    projects = {
        'Qt': base_path / "qt_50projects_irl",
        'Android': base_path / "android_50projects_irl",
        'Chromium': base_path / "chromium_50projects_irl",
        'OpenStack': base_path / "opnestack_50projects_irl_timeseries" / "2x_os",
    }

    all_data = []

    for project_name, project_path in projects.items():
        print(f"\n{'='*60}")
        print(f"分析中: {project_name}")
        print(f"{'='*60}")

        if not project_path.exists():
            print(f"警告: {project_path} が見つかりません")
            continue

        project_data = analyze_developer_acceptance(project_path, project_name)

        if len(project_data) > 0:
            all_data.append(project_data)

            # プロジェクトごとの統計
            stats = {
                '開発者数': len(project_data),
                '平均承諾率': f"{project_data['acceptance_rate'].mean() * 100:.2f}%",
                '中央値承諾率': f"{project_data['acceptance_rate'].median() * 100:.2f}%",
                '標準偏差': f"{project_data['acceptance_rate'].std() * 100:.2f}%",
                '最小承諾率': f"{project_data['acceptance_rate'].min() * 100:.2f}%",
                '最大承諾率': f"{project_data['acceptance_rate'].max() * 100:.2f}%",
                '総リクエスト数': project_data['eval_request_count'].sum(),
                '総承諾数': project_data['eval_accepted_count'].sum(),
                '総拒否数': project_data['eval_rejected_count'].sum(),
            }

            print(f"\n{project_name} の統計:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

            # 承諾率の分布
            print(f"\n承諾率の分布:")
            bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
            labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
            project_data['acceptance_rate_bin'] = pd.cut(
                project_data['acceptance_rate'],
                bins=bins,
                labels=labels,
                include_lowest=True
            )
            distribution = project_data['acceptance_rate_bin'].value_counts().sort_index()
            for bin_label, count in distribution.items():
                percentage = count / len(project_data) * 100
                print(f"  {bin_label}: {count}人 ({percentage:.1f}%)")

        else:
            print(f"{project_name}: データが見つかりませんでした")

    # 全プロジェクトの結果を結合
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)

        # 結果を保存
        output_dir = Path("/Users/kazuki-h/research/multiproject_research/results/developer_acceptance_rate")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 全開発者データを保存
        combined_df.to_csv(output_dir / "all_developers_acceptance.csv", index=False)
        print(f"\n\n{'='*60}")
        print("結果を保存しました:")
        print(f"  {output_dir / 'all_developers_acceptance.csv'}")

        # プロジェクト別のサマリー
        print(f"\n{'='*60}")
        print("プロジェクト別承諾率サマリー")
        print(f"{'='*60}")

        summary_stats = combined_df.groupby('project').agg({
            'acceptance_rate': ['count', 'mean', 'median', 'std', 'min', 'max'],
            'eval_request_count': 'sum',
            'eval_accepted_count': 'sum',
            'eval_rejected_count': 'sum'
        }).round(4)

        summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]
        summary_stats = summary_stats.rename(columns={
            'acceptance_rate_count': 'developer_count',
            'acceptance_rate_mean': 'avg_acceptance_rate',
            'acceptance_rate_median': 'median_acceptance_rate',
            'acceptance_rate_std': 'std_acceptance_rate',
            'acceptance_rate_min': 'min_acceptance_rate',
            'acceptance_rate_max': 'max_acceptance_rate',
            'eval_request_count_sum': 'total_requests',
            'eval_accepted_count_sum': 'total_accepted',
            'eval_rejected_count_sum': 'total_rejected'
        })

        print(summary_stats)
        summary_stats.to_csv(output_dir / "project_acceptance_summary.csv")
        print(f"\nサマリーを保存しました: {output_dir / 'project_acceptance_summary.csv'}")

        # 各プロジェクトの開発者別データを個別保存
        for project_name in combined_df['project'].unique():
            project_df = combined_df[combined_df['project'] == project_name]

            # 承諾率でソート
            project_df_sorted = project_df.sort_values('acceptance_rate', ascending=False)

            # Top 20とBottom 20を表示
            print(f"\n{'='*60}")
            print(f"{project_name} - 承諾率が高い開発者 Top 20")
            print(f"{'='*60}")
            top_20 = project_df_sorted.head(20)[['reviewer_email', 'acceptance_rate',
                                                   'eval_request_count', 'eval_accepted_count',
                                                   'eval_rejected_count', 'train_period', 'eval_period']]
            top_20['acceptance_rate'] = (top_20['acceptance_rate'] * 100).round(2)
            print(top_20.to_string(index=False))

            print(f"\n{project_name} - 承諾率が低い開発者 Bottom 20")
            print(f"{'='*60}")
            bottom_20 = project_df_sorted.tail(20)[['reviewer_email', 'acceptance_rate',
                                                      'eval_request_count', 'eval_accepted_count',
                                                      'eval_rejected_count', 'train_period', 'eval_period']]
            bottom_20['acceptance_rate'] = (bottom_20['acceptance_rate'] * 100).round(2)
            print(bottom_20.to_string(index=False))

            # 個別ファイルに保存
            project_df.to_csv(output_dir / f"{project_name.lower()}_developers.csv", index=False)
            print(f"\n保存: {output_dir / f'{project_name.lower()}_developers.csv'}")


if __name__ == "__main__":
    main()

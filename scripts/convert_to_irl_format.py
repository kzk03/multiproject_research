#!/usr/bin/env python3
"""
CSVデータをマルチプロジェクト対応IRL形式に変換

build_dataset.pyで生成したCSVデータを、
irl_predictor.pyで使用できる形式に変換します。

プロジェクト個別判定 + 複数プロジェクト横断学習:
- 各活動は特定のプロジェクトに紐づく（project_id）
- プロジェクトごとに継続（承諾）を個別判定（label）
- 複数プロジェクトでの活動パターンを学習（is_cross_project, projects）
- プロジェクト間の相互作用を捉える（cross_project_collaboration_score等）
"""

import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_csv_to_irl_format(csv_path: Path, output_path: Path) -> None:
    """
    CSVデータをIRL形式に変換

    Args:
        csv_path: 入力CSVファイルパス
        output_path: 出力JSONファイルパス
    """
    logger.info(f"Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)

    logger.info(f"Total records: {len(df)}")
    logger.info(f"Unique reviewers: {df['reviewer_email'].nunique()}")
    logger.info(f"Unique projects: {df['project'].nunique()}")

    # レビュアーごとにデータを集約
    developer_data = defaultdict(lambda: {
        'developer_id': None,
        'first_seen': None,
        'last_seen': None,
        'changes_authored': 0,
        'changes_reviewed': 0,
        'projects': set(),
        'activity_history': []
    })

    # 時系列順にソート
    df = df.sort_values('request_time')

    logger.info("Converting to IRL format...")
    for idx, row in df.iterrows():
        reviewer = row['reviewer_email']
        project = row['project']
        request_time = row['request_time']

        # 開発者の基本情報を更新
        if developer_data[reviewer]['developer_id'] is None:
            developer_data[reviewer]['developer_id'] = reviewer

        developer_data[reviewer]['projects'].add(project)

        # 初回・最終活動日を更新
        if developer_data[reviewer]['first_seen'] is None:
            developer_data[reviewer]['first_seen'] = request_time
        developer_data[reviewer]['last_seen'] = request_time

        developer_data[reviewer]['changes_reviewed'] += 1

        # 活動履歴を追加
        activity = {
            'type': 'review',
            'developer_id': reviewer,
            'project_id': project,
            'change_id': row['change_id'],
            'timestamp': request_time,
            'request_time': request_time,
            'response_time': row.get('first_response_time', request_time),
            'files_changed': int(row.get('change_files_count', 0)),
            'lines_added': int(row.get('change_insertions', 0)),
            'lines_deleted': int(row.get('change_deletions', 0)),
            'is_cross_project': bool(row.get('is_cross_project', False)),
            'accepted': bool(row['label'] == 1),
            'action_type': 'review',
        }

        developer_data[reviewer]['activity_history'].append(activity)

    # 開発者データを変換
    developers = []
    for dev_id, data in developer_data.items():
        developers.append({
            'developer_id': data['developer_id'],
            'first_seen': data['first_seen'],
            'last_seen': data['last_seen'],
            'changes_authored': data['changes_authored'],
            'changes_reviewed': data['changes_reviewed'],
            'projects': list(data['projects']),
            'activity_history': data['activity_history'],
        })

    # 出力データを構築
    output_data = {
        'developers': developers,
        'projects': df['project'].unique().tolist(),
        'conversion_date': datetime.now().isoformat(),
        'source_file': str(csv_path),
        'statistics': {
            'total_developers': len(developers),
            'total_activities': len(df),
            'total_projects': df['project'].nunique(),
            'date_range': {
                'start': df['request_time'].min(),
                'end': df['request_time'].max(),
            }
        }
    }

    # JSON保存
    logger.info(f"Saving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info("Conversion complete!")
    logger.info(f"  Developers: {len(developers)}")
    logger.info(f"  Activities: {len(df)}")
    logger.info(f"  Projects: {df['project'].nunique()}")


def main():
    parser = argparse.ArgumentParser(
        description='CSVデータをマルチプロジェクト対応IRL形式に変換'
    )
    parser.add_argument('--input', required=True,
                        help='入力CSVファイルパス')
    parser.add_argument('--output', required=True,
                        help='出力JSONファイルパス')

    args = parser.parse_args()

    convert_csv_to_irl_format(Path(args.input), Path(args.output))


if __name__ == '__main__':
    main()

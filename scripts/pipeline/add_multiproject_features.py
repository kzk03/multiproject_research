#!/usr/bin/env python3
"""
既存のデータセットにマルチプロジェクト特徴量を追加するスクリプト

is_cross_project: レビュアーが複数プロジェクトで活動している場合True
reviewer_project_count: レビュアーが参加しているプロジェクト数（その時点まで）
"""

import argparse
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def add_multiproject_features(input_csv: str, output_csv: str):
    """
    既存のレビュー依頼データセットにマルチプロジェクト特徴量を追加

    Args:
        input_csv: 入力CSVファイルパス
        output_csv: 出力CSVファイルパス
    """
    logger.info("=" * 80)
    logger.info("マルチプロジェクト特徴量追加処理")
    logger.info("=" * 80)
    logger.info(f"入力ファイル: {input_csv}")
    logger.info(f"出力ファイル: {output_csv}")

    # データ読み込み
    logger.info("データ読み込み中...")
    df = pd.read_csv(input_csv)
    logger.info(f"読み込み完了: {len(df)}行")

    # 必須カラムのチェック
    required_cols = ['reviewer_email', 'project', 'context_date']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"必須カラムが不足しています: {missing_cols}")

    # context_dateをdatetimeに変換
    df['context_date'] = pd.to_datetime(df['context_date'])

    # 時系列順にソート（重要：過去のデータから順に処理）
    df = df.sort_values('context_date').reset_index(drop=True)

    # 各レビュアーのプロジェクト履歴を追跡
    developer_projects = defaultdict(set)

    # マルチプロジェクト特徴量を格納するリスト
    is_cross_project_list = []
    reviewer_project_count_list = []

    logger.info("マルチプロジェクト特徴量を計算中...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        reviewer = row['reviewer_email']
        project = row['project']

        # この時点でのレビュアーのプロジェクト数
        reviewer_project_count = len(developer_projects[reviewer])

        # クロスプロジェクトフラグ（2つ以上のプロジェクトで活動している場合True）
        is_cross_project = reviewer_project_count > 1

        # リストに追加
        is_cross_project_list.append(is_cross_project)
        reviewer_project_count_list.append(reviewer_project_count)

        # このレビュアーのプロジェクト履歴を更新（次の行のために）
        developer_projects[reviewer].add(project)

    # 新しいカラムを追加
    df['is_cross_project'] = is_cross_project_list
    df['reviewer_project_count'] = reviewer_project_count_list

    logger.info("=" * 80)
    logger.info("特徴量統計:")
    logger.info(f"  is_cross_project=True: {df['is_cross_project'].sum()} ({df['is_cross_project'].mean()*100:.1f}%)")
    logger.info(f"  is_cross_project=False: {(~df['is_cross_project']).sum()} ({(~df['is_cross_project']).mean()*100:.1f}%)")
    logger.info(f"  reviewer_project_count 平均: {df['reviewer_project_count'].mean():.2f}")
    logger.info(f"  reviewer_project_count 最大: {df['reviewer_project_count'].max()}")
    logger.info("=" * 80)

    # 出力
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"出力完了: {output_path}")
    logger.info(f"データセット形状: {df.shape}")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='既存データセットにマルチプロジェクト特徴量を追加',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
    python scripts/pipeline/add_multiproject_features.py \\
        --input data/review_requests_openstack_multi_5y_detail.csv \\
        --output data/review_requests_openstack_multi_5y_with_multiproject.csv
        """
    )

    parser.add_argument('--input', required=True,
                        help='入力CSVファイルパス')
    parser.add_argument('--output', required=True,
                        help='出力CSVファイルパス')

    args = parser.parse_args()

    add_multiproject_features(args.input, args.output)

    logger.info("完了！")


if __name__ == '__main__':
    main()

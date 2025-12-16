#!/usr/bin/env python3
"""
Nova単体プロジェクト用: IRLモデルから特徴量を抽出

状態特徴量: 10次元（マルチプロジェクト特徴量を除外）
行動特徴量: 4次元（cross_project_action_ratioを除外）

マルチプロジェクト対応のextract_state_features.pyから、
マルチプロジェクト特徴量をコメントアウトしたバージョン。
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

# パス設定
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

# train_model.pyのパスを追加
TRAIN_SCRIPTS = ROOT / "scripts" / "train"
if str(TRAIN_SCRIPTS) not in sys.path:
    sys.path.append(str(TRAIN_SCRIPTS))

from review_predictor.model.irl_predictor_nova_single import RetentionIRLSystem
from train_model import (
    extract_evaluation_trajectories,
    load_review_requests
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_features_from_model(
    model_path: Path,
    data_path: Path,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    output_path: Path,
    project_filter: str = None
):
    """
    Nova単体用: IRLモデルから状態10次元・行動4次元の特徴量を抽出

    Args:
        model_path: 学習済みモデルのパス
        data_path: レビューデータCSVのパス
        train_start: 訓練期間開始
        train_end: 訓練期間終了
        eval_start: 評価期間開始
        eval_end: 評価期間終了
        output_path: 出力CSVパス
        project_filter: プロジェクトフィルタ（例: "openstack/nova"）
    """

    logger.info("="*80)
    logger.info("Nova単体用: IRLモデルから特徴量を抽出")
    logger.info("特徴量: 状態10次元 + 行動4次元（マルチプロジェクト特徴量除外）")
    logger.info("="*80)

    # [1] モデル読み込み
    logger.info(f"[1/5] モデルを読み込み: {model_path}")

    # Nova単体は状態10次元・行動4次元で訓練されている
    # モデル読み込み時のconfigも10次元・4次元にする
    config = {
        'state_dim': 10,  # Nova単体: 10次元
        'action_dim': 4,  # Nova単体: 4次元
        'hidden_dim': 128,
        'learning_rate': 0.0001,
        'sequence': True,
        'seq_len': 0,
        'dropout': 0.2
    }

    irl_system = RetentionIRLSystem(config)
    irl_system.network.load_state_dict(torch.load(model_path, map_location='cpu'))
    irl_system.network.eval()

    logger.info("モデル読み込み完了")

    # [2] データ読み込み
    logger.info(f"[2/5] データを読み込み: {data_path}")
    df = load_review_requests(str(data_path))

    # プロジェクトフィルタ
    if project_filter:
        logger.info(f"プロジェクトフィルタ: {project_filter}")
        df = df[df['project'] == project_filter].copy()
        logger.info(f"フィルタ後のレコード数: {len(df)}")

    # [3] 評価用軌跡を抽出
    logger.info(f"[3/5] 評価用軌跡を抽出...")
    logger.info(f"  訓練期間: {train_start} ~ {train_end}")
    logger.info(f"  評価期間: {eval_start} ~ {eval_end}")

    history_window_months = int((train_end - train_start).days / 30)
    months_to_eval = int((eval_start - train_end).days / 30)
    future_window_months = int((eval_end - eval_start).days / 30)

    eval_trajectories = extract_evaluation_trajectories(
        df,
        cutoff_date=train_end,
        history_window_months=history_window_months,
        future_window_start_months=months_to_eval,
        future_window_end_months=months_to_eval + future_window_months,
        min_history_requests=0
    )

    logger.info(f"軌跡数: {len(eval_trajectories)}")

    # [4] 特徴量抽出
    logger.info(f"[4/5] 特徴量を抽出中...")

    all_features = []

    for i, traj in enumerate(eval_trajectories):
        if i % 100 == 0:
            logger.info(f"  処理中: {i}/{len(eval_trajectories)}")

        try:
            # 状態特徴量を抽出
            state = irl_system.extract_developer_state(
                traj['developer'],
                traj['activity_history'],
                traj['context_date']
            )

            # 行動特徴量を抽出
            actions = irl_system.extract_developer_actions(
                traj['activity_history'],
                traj['context_date']
            )

            # 予測確率
            result = irl_system.predict_continuation_probability_snapshot(
                traj['developer'],
                traj['activity_history'],
                traj['context_date']
            )

            # 状態特徴量（10次元のみ抽出）
            # マルチプロジェクト特徴量をコメントアウト
            state_features = {
                'experience_days': state.experience_days,
                'total_changes': state.total_changes,
                'total_reviews': state.total_reviews,
                'recent_activity_frequency': state.recent_activity_frequency,
                'avg_activity_gap': state.avg_activity_gap,
                'activity_trend': state.activity_trend,
                'collaboration_score': state.collaboration_score,
                'code_quality_score': state.code_quality_score,
                'recent_acceptance_rate': state.recent_acceptance_rate,
                'review_load': state.review_load,
                # マルチプロジェクト特徴量（Nova単体では使用しない）
                # 'project_count': state.project_count,
                # 'project_activity_distribution': state.project_activity_distribution,
                # 'main_project_contribution_ratio': state.main_project_contribution_ratio,
                # 'cross_project_collaboration_score': state.cross_project_collaboration_score,
            }

            # 行動特徴量の統計（4次元のみ抽出）
            if actions:
                action_features = {
                    'avg_action_intensity': np.mean([a.intensity for a in actions]),
                    'avg_collaboration': np.mean([a.collaboration for a in actions]),
                    'avg_response_time': np.mean([a.response_time for a in actions]),
                    'avg_review_size': np.mean([a.review_size for a in actions]),
                    # マルチプロジェクト特徴量（Nova単体では使用しない）
                    # 'cross_project_action_ratio': np.mean([1 if a.is_cross_project else 0 for a in actions]),
                }
            else:
                action_features = {
                    'avg_action_intensity': 0.0,
                    'avg_collaboration': 0.0,
                    'avg_response_time': 0.0,
                    'avg_review_size': 0.0,
                    # 'cross_project_action_ratio': 0.0,
                }

            # 予測情報
            prediction_info = {
                'reviewer_email': traj['reviewer'],
                'predicted_prob': result['continuation_probability'],
                'true_label': 1 if traj['future_acceptance'] else 0,
                'history_request_count': traj['history_request_count'],
                'eval_request_count': traj['eval_request_count'],
                'eval_accepted_count': traj['eval_accepted_count'],
                'eval_rejected_count': traj['eval_rejected_count'],
                'context_date': traj['context_date'].strftime('%Y-%m-%d'),
            }

            # 統合
            row = {**prediction_info, **state_features, **action_features}
            all_features.append(row)

        except Exception as e:
            logger.warning(f"軌跡 {i} でエラー: {e}")
            continue

    logger.info(f"特徴量抽出完了: {len(all_features)}件")

    # [5] CSV保存
    logger.info(f"[5/5] CSVに保存: {output_path}")

    df_features = pd.DataFrame(all_features)
    df_features.to_csv(output_path, index=False)

    logger.info("="*80)
    logger.info("抽出完了!")
    logger.info("="*80)
    logger.info(f"出力: {output_path}")
    logger.info(f"レコード数: {len(df_features)}")
    logger.info(f"特徴量次元: {len(df_features.columns) - 8}次元")  # 予測情報8列を除く
    logger.info(f"  状態特徴量: 10次元")
    logger.info(f"  行動特徴量: 4次元")
    logger.info("")
    logger.info("列一覧:")
    for col in df_features.columns:
        logger.info(f"  - {col}")
    logger.info("")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Nova単体用: IRLモデルから特徴量を抽出')
    parser.add_argument('--model', required=True, help='モデルファイルパス (.pt)')
    parser.add_argument('--data', required=True, help='レビューデータCSV')
    parser.add_argument('--train-start', required=True, help='訓練期間開始 (YYYY-MM-DD)')
    parser.add_argument('--train-end', required=True, help='訓練期間終了 (YYYY-MM-DD)')
    parser.add_argument('--eval-start', required=True, help='評価期間開始 (YYYY-MM-DD)')
    parser.add_argument('--eval-end', required=True, help='評価期間終了 (YYYY-MM-DD)')
    parser.add_argument('--output', required=True, help='出力CSVパス')
    parser.add_argument('--project-filter', default=None, help='プロジェクトフィルタ (例: openstack/nova)')

    args = parser.parse_args()

    extract_features_from_model(
        model_path=Path(args.model),
        data_path=Path(args.data),
        train_start=pd.Timestamp(args.train_start),
        train_end=pd.Timestamp(args.train_end),
        eval_start=pd.Timestamp(args.eval_start),
        eval_end=pd.Timestamp(args.eval_end),
        output_path=Path(args.output),
        project_filter=args.project_filter
    )


if __name__ == '__main__':
    main()

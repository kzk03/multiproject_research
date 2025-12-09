#!/usr/bin/env python3
"""
時系列クロス評価（Temporal Cross-Validation）

訓練期間を複数の期間に分割し、各期間で訓練したモデルを
異なる期間で評価してヒートマップで可視化します。

使い方:
    uv run python scripts/analysis/temporal_cross_validation.py \
        --reviews data/multiproject_paper_data.csv \
        --start-date 2021-01-01 \
        --end-date 2024-01-01 \
        --n-folds 6 \
        --output outputs/temporal_cv
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# 既存の訓練スクリプトから関数をインポート
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'train'))
from train_model import extract_review_acceptance_trajectories, extract_evaluation_trajectories

from review_predictor.model.irl_predictor import RetentionIRLSystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_time_folds(
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    n_folds: int = 4
) -> List[Dict[str, pd.Timestamp]]:
    """
    時系列フォールドを作成（ラベル期間のみ分割）

    訓練期間は全フォールドで共通（train_start ~ train_end）
    評価期間（ラベル期間）のみを n_folds 個に分割
    例: 評価12ヶ月を4分割 → 各3ヶ月

    Args:
        train_start: 訓練開始日（全フォールド共通）
        train_end: 訓練終了日（全フォールド共通）
        eval_start: 評価開始日
        eval_end: 評価終了日
        n_folds: フォールド数

    Returns:
        フォールドのリスト
    """
    # 評価期間の月数
    eval_months = (eval_end.year - eval_start.year) * 12 + (eval_end.month - eval_start.month)
    fold_eval_months = eval_months // n_folds

    folds = []
    for i in range(n_folds):
        # 評価期間（ラベル期間）のみ分割
        fold_eval_start = eval_start + pd.DateOffset(months=i * fold_eval_months)
        fold_eval_end = eval_start + pd.DateOffset(months=(i + 1) * fold_eval_months)

        folds.append({
            'fold_id': i,
            'train_start': train_start,  # 全フォールド共通
            'train_end': train_end,      # 全フォールド共通
            'eval_start': fold_eval_start,
            'eval_end': fold_eval_end,
            'name': f'{i*fold_eval_months}-{(i+1)*fold_eval_months}m'
        })

    return folds


def train_on_fold(
    df: pd.DataFrame,
    fold: Dict,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    min_history_events: int = 3
) -> Tuple[RetentionIRLSystem, List[Dict]]:
    """
    フォールドで訓練

    訓練期間は全体で固定（train_start ~ train_end）
    ラベル期間のみfoldで指定された期間を使用
    """

    logger.info(f"訓練期間（特徴量）: {train_start} ~ {train_end} (固定)")
    logger.info(f"ラベル期間: {fold['eval_start']} ~ {fold['eval_end']}")

    # ラベル期間の月数を計算
    eval_months = (fold['eval_end'].year - fold['eval_start'].year) * 12 + \
                  (fold['eval_end'].month - fold['eval_start'].month)

    # train_endからラベル期間開始までの月数（future_window_start_monthsに相当）
    future_start_months = (fold['eval_start'].year - train_end.year) * 12 + \
                          (fold['eval_start'].month - train_end.month)

    # ラベル期間終了までの月数
    future_end_months = future_start_months + eval_months

    logger.info(f"ラベル開始までの月数: {future_start_months}ヶ月")
    logger.info(f"ラベル終了までの月数: {future_end_months}ヶ月")

    # 軌跡抽出（訓練期間は全フォールドで固定）
    trajectories = extract_review_acceptance_trajectories(
        df=df,
        train_start=train_start,
        train_end=train_end,  # 固定の訓練期間終了日
        future_window_start_months=future_start_months,  # ラベル開始までの月数
        future_window_end_months=future_end_months,  # ラベル終了までの月数
        min_history_requests=min_history_events,
        extended_label_window_months=12
    )

    if len(trajectories) == 0:
        logger.warning(f"フォールド {fold['fold_id']}: 訓練データなし")
        return None, []

    # IRLシステム初期化
    config = {
        'state_dim': 14,
        'action_dim': 5,
        'hidden_dim': 128,
        'learning_rate': 0.0003,
        'sequence': True,
        'seq_len': 0,
        'dropout': 0.2,
    }

    irl_system = RetentionIRLSystem(config)

    # 訓練
    logger.info(f"訓練サンプル数: {len(trajectories)}")
    irl_system.train_irl_temporal_trajectories(
        expert_trajectories=trajectories,
        epochs=10
    )

    return irl_system, trajectories


def evaluate_on_fold(
    model: RetentionIRLSystem,
    df: pd.DataFrame,
    eval_fold: Dict,
    train_start: pd.Timestamp,
    min_history_events: int = 3
) -> Dict[str, float]:
    """
    フォールドで評価

    評価期間（eval_fold）で予測性能を測定
    """

    logger.info(f"評価期間: {eval_fold['eval_start']} ~ {eval_fold['eval_end']}")

    # 評価期間の月数を計算
    eval_months = (eval_fold['eval_end'].year - eval_fold['eval_start'].year) * 12 + \
                  (eval_fold['eval_end'].month - eval_fold['eval_start'].month)

    # train_startからの履歴期間（固定：全訓練期間）
    history_months = (eval_fold['eval_start'].year - train_start.year) * 12 + \
                     (eval_fold['eval_start'].month - train_start.month)

    # 将来窓の開始（train_startからの月数ではなく、cutoff_dateからの月数）
    # cutoff_dateは固定の訓練終了日とする
    future_start_months = (eval_fold['eval_start'].year - eval_fold['eval_start'].year) * 12 + \
                          (eval_fold['eval_start'].month - eval_fold['eval_start'].month)

    logger.info(f"履歴期間: {train_start} ~ {eval_fold['eval_start']} ({history_months}ヶ月)")
    logger.info(f"評価期間月数: {eval_months}ヶ月")

    eval_trajectories = extract_evaluation_trajectories(
        df=df,
        cutoff_date=eval_fold['eval_start'],  # 評価期間の開始時点
        history_window_months=history_months,  # 全訓練期間
        future_window_start_months=0,
        future_window_end_months=eval_months,  # 評価期間の長さ
        min_history_requests=min_history_events,
        extended_label_window_months=12
    )

    if len(eval_trajectories) == 0:
        logger.warning(f"評価データなし")
        return {
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'auc_roc': 0.0,
            'count': 0
        }

    # 予測
    predictions = []
    labels = []

    for traj in eval_trajectories:
        try:
            result = model.predict_continuation_probability_snapshot(
                developer=traj['developer_info'],
                activity_history=traj['activity_history'],
                context_date=traj['context_date']
            )

            prob = result['continuation_probability']
            predictions.append(prob)
            labels.append(traj['future_acceptance'])

        except Exception as e:
            logger.warning(f"予測エラー: {e}")
            continue

    if len(predictions) == 0:
        return {
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'auc_roc': 0.0,
            'count': 0
        }

    # 閾値0.5で二値化
    pred_binary = [1 if p >= 0.5 else 0 for p in predictions]

    # メトリクス計算
    f1 = f1_score(labels, pred_binary, zero_division=0)
    precision = precision_score(labels, pred_binary, zero_division=0)
    recall = recall_score(labels, pred_binary, zero_division=0)

    # AUC-ROC
    try:
        auc_roc = roc_auc_score(labels, predictions)
    except:
        auc_roc = 0.0

    logger.info(f"評価結果: F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}, AUC={auc_roc:.3f}, Count={len(predictions)}")

    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc_roc': auc_roc,
        'count': len(predictions),
        'predictions': predictions,
        'labels': labels
    }


def create_heatmap(
    results: np.ndarray,
    fold_names: List[str],
    metric_name: str,
    output_path: Path,
    title: str = None
):
    """ヒートマップ作成"""

    plt.figure(figsize=(12, 10))

    # NaN値をマスク
    masked_results = np.ma.masked_where(results == 0, results)

    sns.heatmap(
        masked_results,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        xticklabels=fold_names,
        yticklabels=fold_names,
        vmin=0,
        vmax=1,
        cbar_kws={'label': metric_name.upper()},
        linewidths=0.5,
        linecolor='gray'
    )

    plt.xlabel('評価期間', fontsize=12, fontweight='bold')
    plt.ylabel('訓練期間', fontsize=12, fontweight='bold')

    if title:
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
    else:
        plt.title(f'時系列クロス評価 - {metric_name.upper()}', fontsize=14, fontweight='bold', pad=20)

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"ヒートマップ保存: {output_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='時系列クロス評価')
    parser.add_argument('--reviews', required=True, help='レビューデータCSV')
    parser.add_argument('--train-start', required=True, help='訓練開始日 (YYYY-MM-DD)')
    parser.add_argument('--train-end', required=True, help='訓練終了日 (YYYY-MM-DD)')
    parser.add_argument('--eval-start', required=True, help='評価開始日 (YYYY-MM-DD)')
    parser.add_argument('--eval-end', required=True, help='評価終了日 (YYYY-MM-DD)')
    parser.add_argument('--n-folds', type=int, default=4, help='フォールド数')
    parser.add_argument('--min-history-events', type=int, default=3, help='最小履歴イベント数')
    parser.add_argument('--output', required=True, help='出力ディレクトリ')

    args = parser.parse_args()

    # 出力ディレクトリ作成
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # データ読み込み
    logger.info(f"データ読み込み: {args.reviews}")
    df = pd.read_csv(args.reviews)
    df['request_time'] = pd.to_datetime(df['request_time'])

    logger.info(f"総レビュー数: {len(df)}")

    # 日付変換
    train_start = pd.to_datetime(args.train_start)
    train_end = pd.to_datetime(args.train_end)
    eval_start = pd.to_datetime(args.eval_start)
    eval_end = pd.to_datetime(args.eval_end)

    # フォールド作成
    logger.info("=" * 80)
    logger.info("時系列フォールド作成")
    logger.info("=" * 80)
    logger.info(f"訓練期間全体: {train_start} ~ {train_end}")
    logger.info(f"評価期間全体: {eval_start} ~ {eval_end}")
    logger.info(f"フォールド数: {args.n_folds}")

    folds = create_time_folds(
        train_start, train_end, eval_start, eval_end, args.n_folds
    )

    for fold in folds:
        logger.info(f"フォールド {fold['fold_id']}: {fold['name']}")
        logger.info(f"  訓練: {fold['train_start']} ~ {fold['train_end']}")
        logger.info(f"  評価: {fold['eval_start']} ~ {fold['eval_end']}")

    # 各フォールドで訓練
    logger.info("=" * 80)
    logger.info("各期間でモデル訓練")
    logger.info("=" * 80)

    models = {}
    for fold in folds:
        logger.info(f"\nフォールド {fold['fold_id']}: {fold['name']}")
        model, train_traj = train_on_fold(
            df, fold, train_start, train_end, args.min_history_events
        )

        if model is not None:
            models[fold['fold_id']] = {
                'model': model,
                'fold': fold,
                'train_count': len(train_traj)
            }

            # モデル保存
            model_path = output_dir / f"model_fold{fold['fold_id']}.pt"
            model.save_model(str(model_path))

    # クロス評価
    logger.info("=" * 80)
    logger.info("クロス評価")
    logger.info("=" * 80)

    n_folds = len(folds)
    results = {
        'f1': np.zeros((n_folds, n_folds)),
        'precision': np.zeros((n_folds, n_folds)),
        'recall': np.zeros((n_folds, n_folds)),
        'auc_roc': np.zeros((n_folds, n_folds)),
        'count': np.zeros((n_folds, n_folds))
    }

    detailed_results = {}

    for train_fold_id, model_info in models.items():
        detailed_results[train_fold_id] = {}

        for eval_fold_id, eval_fold in enumerate(folds):
            logger.info(f"\n訓練: フォールド {train_fold_id} → 評価: フォールド {eval_fold_id}")

            metrics = evaluate_on_fold(
                model_info['model'],
                df,
                eval_fold,
                train_start,
                args.min_history_events
            )

            detailed_results[train_fold_id][eval_fold_id] = metrics

            # 結果マトリクスに格納
            for metric in ['f1', 'precision', 'recall', 'auc_roc', 'count']:
                results[metric][train_fold_id, eval_fold_id] = metrics[metric]

    # 結果保存
    results_path = output_dir / 'temporal_cv_results.json'
    with open(results_path, 'w') as f:
        # NumPy配列をリストに変換
        json_results = {
            'folds': [{'fold_id': f['fold_id'], 'name': f['name']} for f in folds],
            'metrics': {
                metric: results[metric].tolist()
                for metric in ['f1', 'precision', 'recall', 'auc_roc', 'count']
            }
        }
        json.dump(json_results, f, indent=2)

    logger.info(f"結果保存: {results_path}")

    # ヒートマップ作成
    logger.info("=" * 80)
    logger.info("ヒートマップ作成")
    logger.info("=" * 80)

    fold_names = [f['name'] for f in folds]

    for metric in ['f1', 'precision', 'recall', 'auc_roc']:
        heatmap_path = output_dir / f'heatmap_{metric}.png'
        create_heatmap(
            results[metric],
            fold_names,
            metric,
            heatmap_path,
            title=f'時系列クロス評価 - {metric.upper()}\n(行: 訓練期間, 列: 評価期間)'
        )

    # サマリー統計
    logger.info("=" * 80)
    logger.info("サマリー統計")
    logger.info("=" * 80)

    # 対角成分（同一期間）と非対角成分（異なる期間）
    diagonal_f1 = []
    off_diagonal_f1 = []
    future_f1 = []  # 未来の期間での評価
    past_f1 = []  # 過去の期間での評価

    for i in range(n_folds):
        for j in range(n_folds):
            f1 = results['f1'][i, j]
            if f1 > 0:  # 有効なデータがある場合のみ
                if i == j:
                    diagonal_f1.append(f1)
                else:
                    off_diagonal_f1.append(f1)
                    if j > i:
                        future_f1.append(f1)
                    else:
                        past_f1.append(f1)

    if diagonal_f1:
        logger.info(f"同一期間評価 (F1平均): {np.mean(diagonal_f1):.3f} ± {np.std(diagonal_f1):.3f}")
    if off_diagonal_f1:
        logger.info(f"異なる期間評価 (F1平均): {np.mean(off_diagonal_f1):.3f} ± {np.std(off_diagonal_f1):.3f}")
    if future_f1:
        logger.info(f"未来期間での評価 (F1平均): {np.mean(future_f1):.3f} ± {np.std(future_f1):.3f}")
    if past_f1:
        logger.info(f"過去期間での評価 (F1平均): {np.mean(past_f1):.3f} ± {np.std(past_f1):.3f}")

    # 時系列安定性スコア
    if diagonal_f1 and off_diagonal_f1:
        stability_score = np.mean(off_diagonal_f1) / np.mean(diagonal_f1)
        logger.info(f"時系列安定性スコア: {stability_score:.3f}")

    logger.info("=" * 80)
    logger.info("完了！")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
承諾予測のクロス時間評価（既存の継続IRLルートを触らず別ルートで実験）

特徴
- 入力: `data/openstack_20proj_2020_2024_feat.csv`（リクエスト単位、labelが承諾/拒否）
- モデル: 数値特徴のみを使ったロジスティック回帰 + 標準化 + 欠損中央値埋め
- 期間: 訓練0-3m/3-6m/6-9m/9-12m × 評価0-3m..9-12m（訓練開始<=評価開始の10パターン）
- 出力: train_xxx/eval_xxx/metrics.json と matrix_*.csv, summary_statistics.json

使い方（例）:
    uv run python scripts/evaluate/cross_temporal_acceptance_baseline.py \
        --reviews data/openstack_20proj_2020_2024_feat.csv \
        --output results/cross_temporal_acceptance_baseline
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Period:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp


def generate_periods(base_start: pd.Timestamp, total_months: int = 12, step_months: int = 3) -> List[Period]:
    periods: List[Period] = []
    for i in range(0, total_months, step_months):
        start = base_start + pd.DateOffset(months=i)
        end = base_start + pd.DateOffset(months=i + step_months)
        periods.append(Period(name=f"{i}-{i + step_months}m", start=start, end=end))
    return periods


def generate_patterns(train_periods: List[Period], eval_periods: List[Period]) -> List[Tuple[Period, Period]]:
    patterns: List[Tuple[Period, Period]] = []
    for t in train_periods:
        for e in eval_periods:
            # 訓練開始<=評価開始の組み合わせのみ
            if t.start <= e.start:
                patterns.append((t, e))
    return patterns


def select_numeric_features(df: pd.DataFrame, label_col: str = "label") -> Tuple[pd.DataFrame, pd.Series]:
    y = df[label_col].astype(int)
    X = df.select_dtypes(include=["number", "bool"]).copy()
    if label_col in X.columns:
        X = X.drop(columns=[label_col])
    return X, y


def build_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=300,
                    class_weight="balanced",
                    solver="lbfgs",
                    n_jobs=None,
                ),
            ),
        ]
    )


def find_best_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    # F1最大化の閾値を探索（0.0-1.0を201分割）
    best_th = 0.5
    best_f1 = -1.0
    for th in np.linspace(0, 1, 201):
        pred = (prob >= th).astype(int)
        f1 = f1_score(y_true, pred)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
    return float(best_th)


def evaluate(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (prob >= threshold).astype(int)
    return {
        "AUC_ROC": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "AUC_PR": float(average_precision_score(y_true, prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "Precision": float(precision_score(y_true, pred)) if pred.sum() > 0 else float("nan"),
        "Recall": float(recall_score(y_true, pred)) if pred.sum() > 0 else float("nan"),
        "F1": float(f1_score(y_true, pred)) if pred.sum() > 0 else float("nan"),
        "threshold": float(threshold),
    }


def train_and_eval(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> Optional[Dict[str, float]]:
    # ラベルが単一ならスキップ
    if train_df["label"].nunique() < 2 or eval_df["label"].nunique() < 2:
        return None

    X_train, y_train = select_numeric_features(train_df)
    X_eval, y_eval = select_numeric_features(eval_df)

    model = build_model()
    model.fit(X_train, y_train)

    train_prob = model.predict_proba(X_train)[:, 1]
    best_th = find_best_threshold(y_train.values, train_prob)

    eval_prob = model.predict_proba(X_eval)[:, 1]
    metrics = evaluate(y_eval.values, eval_prob, best_th)
    metrics["train_size"] = int(len(train_df))
    metrics["eval_size"] = int(len(eval_df))
    return metrics


def build_matrices(metrics_map: Dict[str, Dict[str, Dict[str, float]]], train_names: List[str], eval_names: List[str]) -> Dict[str, pd.DataFrame]:
    matrices: Dict[str, pd.DataFrame] = {}
    for metric in ["AUC_ROC", "AUC_PR", "Precision", "Recall", "F1"]:
        mat = pd.DataFrame(index=train_names, columns=eval_names, dtype=float)
        for t in train_names:
            for e in eval_names:
                if t in metrics_map and e in metrics_map[t]:
                    mat.loc[t, e] = metrics_map[t][e].get(metric, np.nan)
        matrices[metric] = mat
    return matrices


def main():
    parser = argparse.ArgumentParser(description="承諾予測のクロス時間評価（ロジスティック回帰）")
    parser.add_argument("--reviews", type=str, default="data/openstack_20proj_2020_2024_feat.csv")
    parser.add_argument("--output", type=str, default="results/cross_temporal_acceptance_baseline")
    parser.add_argument("--train-base-start", type=str, default="2021-01-01")
    parser.add_argument("--eval-base-start", type=str, default="2023-01-01")
    parser.add_argument("--total-months", type=int, default=12)
    parser.add_argument("--step-months", type=int, default=3)
    parser.add_argument("--project", type=str, default=None, help="単一プロジェクト名でフィルタ（例: openstack/nova）")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("データを読み込み中...")
    df = pd.read_csv(args.reviews)
    df["request_time"] = pd.to_datetime(df["request_time"])
    if args.project:
        if "project" not in df.columns:
            logger.error("project列が存在しないためフィルタできません")
            return
        before = len(df)
        df = df[df["project"] == args.project].copy()
        logger.info(f"プロジェクトフィルタ: {args.project} ({len(df)}/{before})")

    train_periods = generate_periods(pd.Timestamp(args.train_base_start), args.total_months, args.step_months)
    eval_periods = generate_periods(pd.Timestamp(args.eval_base_start), args.total_months, args.step_months)
    patterns = generate_patterns(train_periods, eval_periods)

    metrics_map: Dict[str, Dict[str, Dict[str, float]]] = {}

    for train_p, eval_p in patterns:
        logger.info("=" * 80)
        logger.info(f"パターン: {train_p.name} → {eval_p.name}")

        train_slice = df[(df["request_time"] >= train_p.start) & (df["request_time"] < train_p.end)].copy()
        eval_slice = df[(df["request_time"] >= eval_p.start) & (df["request_time"] < eval_p.end)].copy()

        if len(train_slice) == 0 or len(eval_slice) == 0:
            logger.warning("データ不足のためスキップ")
            continue

        result = train_and_eval(train_slice, eval_slice)
        if result is None:
            logger.warning("ラベルの多様性不足のためスキップ")
            continue

        # 保存
        train_dir = output_dir / f"train_{train_p.name}"
        eval_dir = train_dir / f"eval_{eval_p.name}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        with open(eval_dir / "metrics.json", "w") as f:
            json.dump(result, f, indent=2)

        metrics_map.setdefault(train_p.name, {})[eval_p.name] = result

    # マトリクスを作成
    train_names = [p.name for p in train_periods]
    eval_names = [p.name for p in eval_periods]
    matrices = build_matrices(metrics_map, train_names, eval_names)
    for metric, mat in matrices.items():
        mat.to_csv(output_dir / f"matrix_{metric}.csv")

    # サマリ
    summary = {metric: float(mat.stack().mean()) for metric, mat in matrices.items() if not mat.stack().empty}
    with open(output_dir / "summary_statistics.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("評価完了")
    logger.info(f"出力ディレクトリ: {output_dir}")


if __name__ == "__main__":
    main()

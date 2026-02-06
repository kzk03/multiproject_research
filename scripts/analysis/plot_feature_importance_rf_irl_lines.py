# -*- coding: utf-8 -*-
"""
RF/IRL の全特徴量（14 個）重要度を期間別に折れ線プロットするスクリプト。
- RF: outputs/rf_nova_cross_eval_unified_rs42/train_<period>/feature_importances.csv
- IRL: results/review_continuation_cross_eval_nova/train_<period>/feature_importance/gradient_importance.json
出力: outputs/rf_vs_irl_nova_summary/features/{rf,irl}/ に PNG/PDF を保存。
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RF_PERIODS = ["0-3m", "3-6m", "6-9m", "9-12m"]

# RF 英語名→日本語名（IRL と揃えた 14 特徴量）
FEATURE_NAME_JA: Dict[str, str] = {
    "experience_days": "経験日数",
    "total_changes": "総レビュー依頼数",
    "total_reviews": "総レビュー数",
    "recent_activity_frequency": "最近の活動頻度",
    "avg_activity_gap": "平均活動間隔",
    "activity_trend": "月次活動変化率",
    "collaboration_score": "協力スコア",
    "code_quality_score": "総承諾率",
    "recent_acceptance_rate": "最近の受諾率",
    "review_load": "レビュー負荷",
    "avg_action_intensity": "レビューファイル数",
    "avg_collaboration": "協力度",
    "avg_response_time": "応答速度",
    "avg_review_size": "レビュー規模（行数）",
}

# 特徴量ごとの固定色（RF/IRL で統一）
FEATURE_COLORS: Dict[str, str] = {
    # 状態特徴量（10次元）
    "経験日数": "#1f77b4",
    "総レビュー依頼数": "#ff7f0e",
    "総レビュー数": "#2ca02c",
    "最近の活動頻度": "#d62728",
    "平均活動間隔": "#9467bd",
    "月次活動変化率": "#8c564b",
    "協力スコア": "#e377c2",
    "総承諾率": "#7f7f7f",
    "最近の受諾率": "#bcbd22",
    "レビュー負荷": "#17becf",
    # 行動特徴量（4次元）
    "レビューファイル数": "#aec7e8",
    "協力度": "#ffbb78",
    "応答速度": "#98df8a",
    "レビュー規模（行数）": "#ff9896",
}


def to_ja(name: str) -> str:
    return FEATURE_NAME_JA.get(name, name)


def load_rf_importances(base_dir: Path) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for period in RF_PERIODS:
        csv_path = base_dir / f"train_{period}" / "feature_importances.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"missing RF file: {csv_path}")
        df = pd.read_csv(csv_path)
        df["period"] = period
        df["feature_ja"] = df["feature"].map(to_ja)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def load_irl_importances(base_dir: Path) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for period in RF_PERIODS:
        json_path = base_dir / f"train_{period}" / "feature_importance" / "gradient_importance.json"
        if not json_path.exists():
            raise FileNotFoundError(f"missing IRL file: {json_path}")
        with open(json_path, "r") as f:
            raw = json.load(f)
        flat: Dict[str, float] = {}
        for section in ["state_importance", "action_importance"]:
            flat.update(raw.get(section, {}))
        df = pd.DataFrame(flat.items(), columns=["feature_ja", "importance"])
        df["period"] = period
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def plot_lines(df: pd.DataFrame, periods: Iterable[str], title: str, out_base: Path, ylabel: str = "重要度") -> None:
    # pivot to features x periods
    pivot = df.pivot_table(index="feature_ja", columns="period", values="importance")
    # sort features by mean importance descending
    pivot["_mean"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("_mean", ascending=False).drop(columns="_mean")

    plt.figure(figsize=(12, 6))
    for feature, row in pivot.iterrows():
        color = FEATURE_COLORS.get(feature, "#000000")
        plt.plot(periods, row[periods], marker="o", label=feature, color=color, linewidth=2)
    plt.xlabel("期間")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle=":", alpha=0.5)
    # 凡例を右側に寄せ、余白を圧縮
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1, fontsize=9)
    plt.tight_layout(rect=(0, 0, 0.92, 1))

    out_base.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_base.with_suffix(".png"), dpi=300)
    plt.savefig(out_base.with_suffix(".pdf"))
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="RF/IRL 特徴量重要度の期間推移を折れ線で出力")
    parser.add_argument(
        "--rf-dir",
        type=Path,
        default=Path("outputs/rf_nova_cross_eval_unified_rs42"),
        help="train_<period>/feature_importances.csv が入った RF 出力ディレクトリ",
    )
    parser.add_argument(
        "--irl-dir",
        type=Path,
        default=Path("results/review_continuation_cross_eval_nova"),
        help="IRL 出力ディレクトリ (train_<period>/feature_importance/gradient_importance.json を含む)",
    )
    parser.add_argument(
        "--rf-out",
        type=Path,
        default=Path("outputs/rf_vs_irl_nova_summary/features/rf/rf_feature_importance_over_periods_jp"),
        help="RF 折れ線の出力ベースパス (拡張子なしで指定)",
    )
    parser.add_argument(
        "--irl-out",
        type=Path,
        default=Path("outputs/rf_vs_irl_nova_summary/features/irl/irl_feature_importance_transition_jp"),
        help="IRL 折れ線の出力ベースパス (拡張子なしで指定)",
    )
    args = parser.parse_args()

    # RF
    rf_df = load_rf_importances(args.rf_dir)
    plot_lines(rf_df, RF_PERIODS, "RF: 特徴量重要度（期間別, Gini）", args.rf_out, ylabel="Gini 重要度")

    # IRL (14特徴量, 勾配ベース)
    irl_df = load_irl_importances(args.irl_dir)
    plot_lines(irl_df, RF_PERIODS, "IRL: 特徴量重要度（期間別, 勾配ベース）", args.irl_out, ylabel="重要度")


if __name__ == "__main__":
    main()

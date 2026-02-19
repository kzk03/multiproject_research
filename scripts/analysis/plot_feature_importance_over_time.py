#!/usr/bin/env python3
"""期間別の特徴量重要度を折れ線で描画するスクリプト。

- 入力: train_<period>/feature_importances.csv を periods (0-3m, 3-6m, 6-9m, 9-12m) から読み込む
- 出力: 折れ線図 (PNG/PDF) を output-dir 直下に保存
"""
import argparse
from pathlib import Path

import japanize_matplotlib  # noqa: F401  # 日本語ラベル用
import matplotlib.pyplot as plt
import pandas as pd

PERIODS = ["0-3m", "3-6m", "6-9m", "9-12m"]

FEATURE_NAME_JA = {
    "avg_action_intensity": "強度（ファイル数）",
    "avg_activity_gap": "平均活動間隔",
    "avg_collaboration": "協力度",
    "avg_review_size": "レビュー規模",
    "code_quality_score": "コード品質",
    "experience_days": "経験日数",
    "recent_activity_frequency": "最近の活動頻度",
    "review_load": "レビュー負荷",
    "total_changes": "総レビュー依頼数",
    "total_reviews": "総レビュー数",
}


def _to_ja(name: str) -> str:
    return FEATURE_NAME_JA.get(name, name)


def load_importances(output_dir: Path) -> pd.DataFrame:
    rows = []
    for period in PERIODS:
        csv_path = output_dir / f"train_{period}" / "feature_importances.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing {csv_path}")
        df = pd.read_csv(csv_path)
        df["period"] = period
        df["feature_ja"] = df["feature"].map(_to_ja)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def plot_lines(df: pd.DataFrame, output_dir: Path, top_n: int):
    # pick top_n features by mean importance across periods
    top_features = (
        df.groupby("feature_ja")["importance"].mean()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )
    subset = df[df["feature_ja"].isin(top_features)]
    pivot = subset.pivot_table(index="feature_ja", columns="period", values="importance")

    plt.figure(figsize=(10, 5))
    for feature, row in pivot.iterrows():
        plt.plot(PERIODS, row[PERIODS], marker="o", label=feature)
    plt.xlabel("訓練期間")
    plt.ylabel("Gini 重要度")
    plt.title("特徴量重要度（期間別）")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.0))
    plt.tight_layout(rect=(0, 0, 0.88, 1))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_base = output_dir / "feature_importance_over_periods"
    plt.savefig(out_base.with_suffix(".png"), dpi=200)
    plt.savefig(out_base.with_suffix(".pdf"))
    plt.close()
    return out_base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory that contains train_<period>/feature_importances.csv",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top features to plot",
    )
    args = parser.parse_args()

    df = load_importances(args.output_dir)
    out_path = plot_lines(df, args.output_dir, args.top_n)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

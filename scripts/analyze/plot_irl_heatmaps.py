# -*- coding: utf-8 -*-
"""
ヒートマップ作成用スクリプト（IRL Nova 行列を結合して描画）
- 入力: performance/irl/ 配下の matrix_*.csv（AUC_ROC, F1, PRECISION, RECALL）
- 出力: heatmap_{AUC_ROC,F1,PRECISION,RECALL}.png と heatmap_4_metrics.png
- 利用例: uv run scripts/analyze/plot_irl_heatmaps.py --input-dir outputs/rf_vs_irl_nova_summary/performance/irl --output-dir outputs/rf_vs_irl_nova_summary/performance/irl
"""

import argparse
import pathlib
from typing import Dict

import japanize_matplotlib  # noqa: F401  # フォントを日本語対応にするだけで利用しない
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

MetricNameMap = {
    "AUC_ROC": "AUC-ROC",
    "F1": "F1",
    "PRECISION": "適合率",
    "RECALL": "再現率",
}

PanelTitle = "IRL: クロス評価ヒートマップ"

FileNameMap = {
    "AUC_ROC": "irl_matrix_AUC_ROC.csv",
    "F1": "irl_matrix_F1.csv",
    "PRECISION": "matrix_PRECISION.csv",
    "RECALL": "matrix_RECALL.csv",
}


def compute_global_range(matrices: Dict[str, pd.DataFrame]) -> tuple[float, float]:
    values = []
    for df in matrices.values():
        values.append(df.values.flatten())
    arr = np.concatenate(values)
    arr = arr[~np.isnan(arr)]
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmin == vmax:
        vmax = vmin + 1e-6
    return vmin, vmax


def load_matrix(csv_path: pathlib.Path) -> pd.DataFrame:
    """CSV を DataFrame として読み込み、数値以外を NaN に揃える。"""
    df = pd.read_csv(csv_path, index_col=0)
    df.columns = [c[5:] if isinstance(c, str) and c.startswith("eval_") else c for c in df.columns]
    df = df.replace({"": np.nan})
    return df.astype(float)


def plot_heatmap(df: pd.DataFrame, title: str, out_path: pathlib.Path, vmin: float, vmax: float) -> None:
    """単一指標のヒートマップを描画して保存。"""
    plt.figure(figsize=(6, 5))
    df_t = df.T  # 評価期間を縦軸に置く
    mask = df_t.isna()
    ax = sns.heatmap(
        df_t,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        mask=mask,
        cbar_kws={"label": title},
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("学習期間")
    ax.set_ylabel("評価期間")
    ax.invert_yaxis()  # 0-3m を下に揃える
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".png"), dpi=300)
    plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()


def plot_four_panel(matrices: Dict[str, pd.DataFrame], out_path: pathlib.Path, vmin: float, vmax: float) -> None:
    """4 指標のヒートマップを 2x2 でまとめて保存。"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    # 左上: PRECISION, 右上: RECALL, 左下: F1, 右下: AUC_ROC
    items = ["PRECISION", "RECALL", "F1", "AUC_ROC"]
    for ax, metric in zip(axes.flat, items):
        df = matrices[metric].T  # 評価期間を縦軸に置く
        mask = df.isna()
        sns.heatmap(
            df,
            annot=True,
            fmt=".3f",
            cmap="Blues",
            mask=mask,
            ax=ax,
            cbar_kws={"label": MetricNameMap[metric]},
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(MetricNameMap[metric])
        ax.set_xlabel("学習期間")
        ax.set_ylabel("評価期間")
        ax.invert_yaxis()
    fig.suptitle(PanelTitle, fontsize=14, y=0.98)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(out_path.with_suffix(".png"), dpi=300)
    plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="IRL Nova 行列からヒートマップを生成する")
    parser.add_argument(
        "--input-dir",
        type=pathlib.Path,
        default=pathlib.Path("outputs/rf_vs_irl_nova_summary/performance/irl"),
        help="matrix_*.csv が置いてあるディレクトリ",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("outputs/rf_vs_irl_nova_summary/performance/irl"),
        help="画像を書き出すディレクトリ (PNG/PDF 両方保存)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    matrices: Dict[str, pd.DataFrame] = {}
    for metric in ["AUC_ROC", "F1", "PRECISION", "RECALL"]:
        csv_path = args.input_dir / FileNameMap[metric]
        matrices[metric] = load_matrix(csv_path)

    vmin, vmax = compute_global_range(matrices)

    for metric, df in matrices.items():
        out_base = args.output_dir / f"heatmap_{metric}"
        plot_heatmap(df, MetricNameMap[metric], out_base, vmin, vmax)

    # 4面まとめは IRLheatmap.{png,pdf}
    plot_four_panel(matrices, args.output_dir / "IRLheatmap", vmin, vmax)


if __name__ == "__main__":
    main()

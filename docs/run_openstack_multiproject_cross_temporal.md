# OpenStack マルチプロジェクト IRL クロス時間評価 手順

## 目的

`data/openstack_multiproject_2020_2024.csv` を用いて、3 ヶ月刻み・訓練期間<=評価期間の全 10 パターンで IRL クロス時間評価を実行し、メトリクスマトリクスとヒートマップを生成する。

## 前提

- `uv` が利用可能で、`pyproject.toml` に基づき依存が解決できること。
- 入力データ: `data/openstack_multiproject_2020_2024.csv` が存在すること。
- 作業ディレクトリ: リポジトリルート `/Users/kazuki-h/research/multiproject_research`。

## 出力

- 出力先: `results/cross_temporal_openstack_multiproject_2020_2024/`
- 生成物: `matrix_*.csv`, `summary_statistics.json`, `train_<期間>/eval_<期間>/` 配下の `metrics.json` と `predictions.csv`, `heatmaps/*.png`

## 評価パターン（10 パターン）

- 0-3m → 0-3m, 3-6m, 6-9m, 9-12m
- 3-6m → 3-6m, 6-9m, 9-12m
- 6-9m → 6-9m, 9-12m
- 9-12m → 9-12m

## 期間設定（既存 result と同一）

- 訓練ベース開始: 2021-01-01（0-3m: 2021-01-01 ～ 2021-04-01, …, 9-12m: ～ 2022-01-01）
- 評価ベース開始: 2023-01-01（0-3m: 2023-01-01 ～ 2023-04-01, …, 9-12m: ～ 2024-01-01）
- 総期間: 12 ヶ月（3 ヶ月 ×4 区間）
- エポック: 20（必要に応じて調整可）

## 実行コマンド

以下をリポジトリルートで実行する。

```bash
uv run python scripts/train/train_cross_temporal_multiproject.py \
  --reviews data/openstack_multiproject_2020_2024.csv \
  --train-base-start 2021-01-01 \
  --eval-base-start 2023-01-01 \
  --total-months 12 \
  --output results/cross_temporal_openstack_multiproject_2020_2024 \
  --epochs 20

uv run python scripts/analysis/create_cross_temporal_heatmaps.py \
  --input results/cross_temporal_openstack_multiproject_2020_2024
```

## 実行時間目安

- マルチプロジェクト設定（CPU のみ）で数時間程度。GPU 使用時は半減程度。

## 実行後の確認

- `results/cross_temporal_openstack_multiproject_2020_2024/matrix_AUC_ROC.csv` などのマトリクス
- `results/cross_temporal_openstack_multiproject_2020_2024/heatmaps/heatmap_4_metrics.png`
- 各 `train_<期間>/eval_<期間>/metrics.json` と `predictions.csv`

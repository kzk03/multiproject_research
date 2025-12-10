# クロス時間評価ガイド

## 概要

このガイドでは、複数プロジェクト対応の IRL モデルを用いたクロス時間評価の実行方法を説明します。

### 評価パターン

訓練期間 ≤ 評価期間の制約で、3 ヶ月間隔の全 10 パターンを評価:

```
訓練期間    評価期間
0-3m    →  0-3m, 3-6m, 6-9m, 9-12m    (4パターン)
3-6m    →  3-6m, 6-9m, 9-12m          (3パターン)
6-9m    →  6-9m, 9-12m                (2パターン)
9-12m   →  9-12m                      (1パターン)
-------------------------------------------
合計: 10パターン
```

## 実行手順

### 1. データセット構築

まず、レビュー依頼データセットを構築します:

```bash
# 単一プロジェクト（例: OpenStack Nova）
uv run python scripts/pipeline/build_dataset.py \
  --gerrit-url https://review.opendev.org \
  --project openstack/nova \
  --start-date 2021-01-01 \
  --end-date 2024-01-01 \
  --output data/nova_reviews.csv

# 複数プロジェクト
uv run python scripts/pipeline/build_dataset.py \
  --gerrit-url https://review.opendev.org \
  --project openstack/nova openstack/neutron openstack/cinder \
  --start-date 2021-01-01 \
  --end-date 2024-01-01 \
  --output data/openstack_multi_reviews.csv
```

### 2. クロス時間評価の実行

全 10 パターンの訓練・評価を自動実行:

```bash
# 単一プロジェクト（既存結果と同じ期間設定）
uv run python scripts/train/train_cross_temporal_multiproject.py \
  --reviews data/nova_reviews.csv \
  --train-base-start 2021-01-01 \
  --eval-base-start 2023-01-01 \
  --total-months 12 \
  --output results/cross_temporal_nova \
  --project openstack/nova \
  --epochs 20

# 複数プロジェクト
uv run python scripts/train/train_cross_temporal_multiproject.py \
  --reviews data/openstack_multi_reviews.csv \
  --train-base-start 2021-01-01 \
  --eval-base-start 2023-01-01 \
  --total-months 12 \
  --output results/cross_temporal_multiproject \
  --epochs 20
```

#### パラメータ説明

- `--reviews`: レビュー依頼 CSV ファイルのパス
- `--train-base-start`: 訓練期間のベース開始日（YYYY-MM-DD、デフォルト: 2021-01-01）
- `--eval-base-start`: 評価期間のベース開始日（YYYY-MM-DD、デフォルト: 2023-01-01）
- `--total-months`: 総期間（月数、デフォルト 12 ヶ月）
- `--output`: 出力ディレクトリ
- `--project`: プロジェクト名（省略時は全プロジェクト）
- `--epochs`: 訓練エポック数（デフォルト 20）
- `--min-history-events`: 最小履歴イベント数（デフォルト 3）

**実際の日付**:

```
訓練期間（2021-01-01起点）:
  0-3m:  2021-01-01 ～ 2021-04-01
  3-6m:  2021-04-01 ～ 2021-07-01
  6-9m:  2021-07-01 ～ 2021-10-01
  9-12m: 2021-10-01 ～ 2022-01-01

評価期間（2023-01-01起点）:
  0-3m:  2023-01-01 ～ 2023-04-01
  3-6m:  2023-04-01 ～ 2023-07-01
  6-9m:  2023-07-01 ～ 2023-10-01
  9-12m: 2023-10-01 ～ 2024-01-01
```

これは既存の `results/review_continuation_cross_eval_nova/` と同じ期間設定です。

### 3. ヒートマップの作成

評価結果からヒートマップを生成:

```bash
uv run python scripts/analysis/create_cross_temporal_heatmaps.py \
  --input results/cross_temporal_nova
```

## 出力ディレクトリ構造

```
results/cross_temporal_nova/
├── README.md                          # このファイル
├── matrix_AUC_ROC.csv                 # AUC-ROCマトリクス
├── matrix_AUC_PR.csv                  # AUC-PRマトリクス
├── matrix_PRECISION.csv               # Precisionマトリクス
├── matrix_RECALL.csv                  # Recallマトリクス
├── matrix_f1_score.csv                # F1 Scoreマトリクス
├── summary_statistics.json            # サマリー統計
│
├── heatmaps/                          # ヒートマップ
│   ├── heatmap_4_metrics.png          # 4メトリクス統合
│   ├── heatmap_AUC_ROC.png            # AUC-ROC
│   ├── heatmap_AUC_PR.png             # AUC-PR
│   ├── heatmap_PRECISION.png          # Precision
│   ├── heatmap_RECALL.png             # Recall
│   └── heatmap_f1_score.png           # F1 Score
│
└── train_<期間>/                      # 各訓練期間のディレクトリ
    ├── irl_model.pt                   # 訓練済みモデル
    ├── optimal_threshold.json         # 最適閾値情報
    │
    └── eval_<期間>/                   # 各評価期間の結果
        ├── metrics.json               # 評価メトリクス
        └── predictions.csv            # 予測結果
```

## メトリクスマトリクスの見方

例: `matrix_AUC_ROC.csv`

```
        0-3m    3-6m    6-9m    9-12m
0-3m   0.717   0.823   0.910*  0.734
3-6m    NaN    0.820   0.894   0.802
6-9m    NaN     NaN    0.785   0.832
9-12m   NaN     NaN     NaN    0.693
```

- **行**: 訓練期間（モデルを訓練したデータ期間）
- **列**: 評価期間（モデルを評価したデータ期間）
- **NaN**: 訓練期間 > 評価期間のため評価なし
- **対角線**: 同一期間での評価
- **オフ対角**: クロス評価（汎化性能）

## 主要な発見の例（参考）

### 予測精度

```
平均AUC-ROC: 0.754  （優れた予測）
最高AUC-ROC: 0.910  （極めて優秀）
平均AUC-PR:  0.656  （実用的）
Precision:   0.778  （推薦の78%が的中）
```

### 最適設定

```
訓練期間: 3-6ヶ月   → 安定した性能
評価期間: 6-9ヶ月   → 予測しやすい期間
最高組合: 0-3m→6-9m → クロス評価で高性能
```

## 実験設定の詳細

### データ期間

- **全体期間**: 36 ヶ月（3 年間）
- **訓練/評価分割**: 各パターンで異なる
- **将来窓**: 評価期間に応じて調整

### モデル設定

- **アーキテクチャ**: IRL + LSTM
- **状態特徴量**: 14 次元（マルチプロジェクト対応）
  - 経験日数、総コミット数、総レビュー数
  - 最近の活動頻度、平均活動間隔
  - 活動トレンド、協力度、コード品質
  - 最近の受諾率、レビュー負荷
  - **プロジェクト数、活動分散度、メイン貢献率、横断協力度** (NEW)
- **行動特徴量**: 5 次元（マルチプロジェクト対応）
  - 行動強度、協力度、応答速度、レビュー規模
  - **クロスプロジェクトフラグ** (NEW)
- **隠れ層**: 128 ユニット
- **Dropout**: 0.2

### ラベリング

- **正例**: 評価期間内にレビュー承諾
- **負例 1**: 評価期間内に依頼あり・承諾なし（重み 1.0）
- **負例 2**: 拡張期間に依頼あり・承諾なし（重み 0.1）
- **除外**: 拡張期間まで依頼なし

## トラブルシューティング

### データ不足エラー

```
エラー: 訓練用軌跡が抽出できませんでした
```

**対処法**:

- `--min-history-events` を減らす（デフォルト 3 → 2）
- データ期間を延長する
- プロジェクトを追加する

### メモリ不足

```
エラー: CUDA out of memory
```

**対処法**:

- バッチサイズを減らす（コード内で調整）
- `--hidden-dim` を減らす（128 → 64）
- CPU で実行する（自動的にフォールバック）

### 性能が低い

```
AUC-ROC < 0.6
```

**対処法**:

- `--epochs` を増やす（20 → 50）
- データの質を確認する
- 特徴量エンジニアリングを見直す

## カスタマイズ

### 期間間隔の変更

3 ヶ月以外の間隔にする場合:

```python
# scripts/train/train_cross_temporal_multiproject.py の修正
# 170行目付近
for i in range(0, total_months, 3):  # 3 → 任意の値
```

### メトリクスの追加

新しいメトリクスを追加する場合:

```python
# scripts/train/train_cross_temporal_multiproject.py の修正
# 190行目付近
metrics_names = ['AUC_ROC', 'AUC_PR', 'PRECISION', 'RECALL', 'f1_score', 'NEW_METRIC']
```

### 特徴量の追加

新しい特徴量を追加する場合:

1. `src/review_predictor/model/irl_predictor.py` を編集
2. `extract_developer_state()` または `extract_developer_actions()` に特徴量を追加
3. `state_dim` または `action_dim` を更新
4. 再訓練を実行

## よくある質問

### Q1: なぜ訓練期間 ≤ 評価期間なのか？

**A**: 過去のデータで訓練したモデルが将来を予測するという現実的な設定を反映するためです。逆（将来のデータで訓練して過去を予測）は実運用では不可能です。

### Q2: 全 16 パターン（4×4）ではなく 10 パターンなのか？

**A**: 訓練期間 > 評価期間のパターン（6 パターン）は現実的でないため除外しています。

### Q3: 複数プロジェクトの利点は？

**A**:

- より多くのデータでモデルを訓練できる
- プロジェクト横断的なパターンを学習できる
- 新しいプロジェクトへの転移学習が可能

### Q4: 実行時間はどのくらい？

**A**:

- 単一プロジェクト（Nova）: 約 2-3 時間（10 パターン）
- 複数プロジェクト（3 つ）: 約 5-7 時間（10 パターン）
- GPU 使用時は半分程度に短縮

### Q5: モデルを他のプロジェクトに適用できるか？

**A**: 可能ですが、以下に注意:

- 再訓練が推奨（プロジェクトごとの特性を学習）
- 少なくとも評価データで性能を確認
- クロス評価で汎化性能を検証

## 次のステップ

1. **結果の分析**

   - ヒートマップを確認
   - 最高性能のパターンを特定
   - サマリー統計を確認

2. **モデルの改善**

   - ハイパーパラメータチューニング
   - 特徴量エンジニアリング
   - アンサンブル学習

3. **実運用への展開**
   - レビュアー推薦システムの構築
   - 離脱リスク検出の実装
   - 定期的なモデル更新の自動化

## 参考資料

- [Review Continuation Prediction IRL](../results/review_continuation_cross_eval_nova/README.md)
- [Temporal IRL Documentation](../README_TEMPORAL_IRL.md)
- [IRL Predictor Implementation](../src/review_predictor/model/irl_predictor.py)

## ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。

---

**作成日**: 2024-12-08
**更新日**: 2024-12-08
**バージョン**: 1.0.0

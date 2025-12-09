# クロス時間評価 - クイックスタート

## 概要

複数プロジェクト対応のIRLで、0-3mの期間で訓練し、0-3m, 3-6m, 6-9m, 9-12mを予測する全10パターンのクロス評価とヒートマップを作成します。

## 評価パターン（全10パターン）

訓練期間 ≤ 評価期間の制約:

```
訓練 → 評価
─────────────
0-3m → 0-3m  ✓
0-3m → 3-6m  ✓
0-3m → 6-9m  ✓
0-3m → 9-12m ✓
3-6m → 3-6m  ✓
3-6m → 6-9m  ✓
3-6m → 9-12m ✓
6-9m → 6-9m  ✓
6-9m → 9-12m ✓
9-12m → 9-12m ✓
```

## 1行で実行

```bash
# 自動パイプライン（最も簡単）
./scripts/run_cross_temporal_evaluation.sh
```

デフォルト設定（既存のresults/review_acceptance_cross_eval_nova/と同じ期間）:
- レビューCSV: `data/review_requests_openstack_multi_5y_detail.csv`
- 訓練期間ベース: `2021-01-01` → 2021-01-01 ～ 2022-01-01（12ヶ月）
- 評価期間ベース: `2023-01-01` → 2023-01-01 ～ 2024-01-01（12ヶ月）
- 総期間: 12ヶ月（4期間 × 3ヶ月）
- 出力: `results/cross_temporal_multiproject`
- プロジェクト: 全プロジェクト
- エポック数: 20

実際の日付:
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

## カスタム設定で実行

```bash
# 引数で設定を変更
./scripts/run_cross_temporal_evaluation.sh \
  data/nova_reviews.csv \
  2021-01-01 \
  2023-01-01 \
  12 \
  results/cross_temporal_nova \
  "openstack/nova" \
  20
```

引数の順序:
1. レビューCSVファイル
2. 訓練期間ベース開始日
3. 評価期間ベース開始日
4. 総期間（月数）
5. 出力ディレクトリ
6. プロジェクト名（空=""で全プロジェクト）
7. エポック数

## 手動で実行（詳細制御）

### ステップ1: データセット構築

```bash
# 単一プロジェクト
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

### ステップ2: クロス時間評価

```bash
# 全10パターンの訓練・評価
uv run python scripts/train/train_cross_temporal_multiproject.py \
  --reviews data/openstack_multi_reviews.csv \
  --train-base-start 2021-01-01 \
  --eval-base-start 2023-01-01 \
  --total-months 12 \
  --output results/cross_temporal_multiproject \
  --epochs 20
```

### ステップ3: ヒートマップ作成

```bash
# ヒートマップと統計を生成
uv run python scripts/analysis/create_cross_temporal_heatmaps.py \
  --input results/cross_temporal_multiproject
```

## 出力の確認

```bash
# ディレクトリ構造を確認
tree results/cross_temporal_multiproject

# AUC-ROCマトリクスを表示
cat results/cross_temporal_multiproject/matrix_AUC_ROC.csv

# サマリー統計を表示
cat results/cross_temporal_multiproject/summary_statistics.json

# ヒートマップを開く
open results/cross_temporal_multiproject/heatmaps/heatmap_4_metrics.png
```

## 出力ファイル一覧

```
results/cross_temporal_multiproject/
├── matrix_AUC_ROC.csv              # AUC-ROCマトリクス（10パターン）
├── matrix_AUC_PR.csv               # AUC-PRマトリクス
├── matrix_PRECISION.csv            # Precisionマトリクス
├── matrix_RECALL.csv               # Recallマトリクス
├── matrix_f1_score.csv             # F1 Scoreマトリクス
├── summary_statistics.json         # サマリー統計
│
├── heatmaps/                       # ヒートマップ画像
│   ├── heatmap_4_metrics.png       # 4メトリクス統合（推奨）★
│   ├── heatmap_AUC_ROC.png
│   ├── heatmap_AUC_PR.png
│   ├── heatmap_PRECISION.png
│   ├── heatmap_RECALL.png
│   └── heatmap_f1_score.png
│
└── train_<期間>/                   # 各訓練期間のディレクトリ
    ├── irl_model.pt                # 訓練済みモデル
    ├── optimal_threshold.json      # 最適閾値
    └── eval_<期間>/                # 各評価期間の結果
        ├── metrics.json            # 評価メトリクス
        └── predictions.csv         # 予測結果
```

## マトリクスの例

```csv
matrix_AUC_ROC.csv:
        0-3m    3-6m    6-9m    9-12m
0-3m   0.717   0.823   0.910   0.734
3-6m    NaN    0.820   0.894   0.802
6-9m    NaN     NaN    0.785   0.832
9-12m   NaN     NaN     NaN    0.693
```

- **行**: 訓練期間
- **列**: 評価期間
- **NaN**: 訓練期間 > 評価期間（評価なし）
- **対角線**: 同一期間での評価
- **オフ対角**: クロス評価（汎化性能）

## 実行時間の目安

| 設定 | 時間 |
|------|------|
| 単一プロジェクト（Nova） | 2-3時間 |
| 複数プロジェクト（3つ） | 5-7時間 |
| GPU使用時 | 半分程度に短縮 |

## トラブルシューティング

### データファイルが見つからない

```
エラー: レビューCSVファイルが見つかりません
```

**対処法**: まずデータセットを構築してください:

```bash
uv run python scripts/pipeline/build_dataset.py \
  --gerrit-url https://review.opendev.org \
  --project openstack/nova \
  --start-date 2021-01-01 \
  --end-date 2024-01-01 \
  --output data/nova_reviews.csv
```

### 軌跡が抽出できない

```
エラー: 訓練用軌跡が抽出できませんでした
```

**対処法**: 最小履歴イベント数を減らす:

```bash
uv run python scripts/train/train_cross_temporal_multiproject.py \
  ... \
  --min-history-events 2  # デフォルト3→2
```

### メモリ不足

```
エラー: CUDA out of memory
```

**対処法**:
1. CPUで実行（自動フォールバック）
2. hidden_dimを減らす（コード内で128→64に変更）

## 次のステップ

1. **結果の分析**
   - ヒートマップで最高性能のパターンを特定
   - サマリー統計で全体的な傾向を確認

2. **詳細ドキュメント**
   - [クロス時間評価ガイド](docs/cross_temporal_evaluation_guide.md)
   - [既存結果の例](results/review_acceptance_cross_eval_nova/README.md)

3. **カスタマイズ**
   - ハイパーパラメータの調整
   - 特徴量の追加
   - 評価期間の変更

## よくある質問

**Q: なぜ10パターンなのか？**

A: 訓練期間 ≤ 評価期間の制約により、4+3+2+1 = 10パターンになります。全16パターン（4×4）ではありません。

**Q: 複数プロジェクトの利点は？**

A: より多くのデータで訓練でき、プロジェクト横断的なパターンを学習できます。新しいプロジェクトへの転移も可能です。

**Q: 実運用での使用方法は？**

A: 最高性能のモデルを使用してレビュアー推薦や離脱リスク検出を実装できます。詳細は[ガイド](docs/cross_temporal_evaluation_guide.md)を参照してください。

## サポート

問題が発生した場合:
1. [トラブルシューティング](#トラブルシューティング)を確認
2. [詳細ガイド](docs/cross_temporal_evaluation_guide.md)を参照
3. GitHubでIssueを作成

---

**作成日**: 2024-12-08
**バージョン**: 1.0.0

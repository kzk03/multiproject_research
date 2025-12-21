# Nova Single Project: Random Forest vs IRL 比較結果

## ⚠️ 重要な更新（2025-12-21）

**データ不整合が発見され、分析を全面的に訂正しました。**

- **旧分析**: 誤ったRFデータ（2020-2021年）を使用していた
- **新分析**: 正しい期間（2023年）で再比較
- **結論の変更**: IRLが優秀 → **RFがIRLを上回る**

詳細は [DATA_INCONSISTENCY_REPORT.md](DATA_INCONSISTENCY_REPORT.md) を参照してください。

---

## ディレクトリ概要

このディレクトリは、OpenStack Novaプロジェクトの単一プロジェクトにおけるレビュー継続予測について、**Random Forest (RF)** と **Inverse Reinforcement Learning (IRL)** の性能比較結果をまとめたものです。

## 📋 必読ドキュメント

### 🎯 [CORRECTED_ANALYSIS.md](CORRECTED_ANALYSIS.md) ⭐⭐⭐ 最優先（NEW!）
**期間を揃えた正しい比較結果**（30-45分で読める）

**内容**:
- 正しい期間（2023年7-10月）での比較結果
- RF vs IRL: 80% vs 67%（RFが優位）
- データ不整合の詳細説明
- 4つのケーススタディ（両モデル正解、IRLのみ正解、RFのみ正解、両モデル不正解）
- john.garbutt@stackhpc.comケースの真相
- 実務への推奨事項

**主要な発見**:
```
正しい比較結果（6-9m期間）:
  RF正解率:  80.0% (12/15)
  IRL正解率: 66.7% (10/15)

カテゴリ分布:
  両モデル正解:    8件（53.3%）
  IRLのみ正解:     2件（13.3%）
  RFのみ正解:      4件（26.7%） ← 新発見
  両モデル不正解:  1件（6.7%）
```

**こんな人におすすめ**:
- **最新の正しい結果を知りたい** ← 最重要
- データ不整合の詳細を理解したい
- どのモデルを使うべきか判断したい
- 具体的なケーススタディを見たい

---

### 🔍 [DATA_INCONSISTENCY_REPORT.md](DATA_INCONSISTENCY_REPORT.md) ⭐⭐ 経緯の理解
**データ不整合の調査レポート**（20-30分）

**内容**:
- john.garbutt@stackhpc.comケースの異常発見
- 旧RFデータ（2020-2021年）vs 新RFデータ（2023年）
- ラベル不一致率26.7%の原因
- 訂正前後の比較
- 今後の対応策

**こんな人におすすめ**:
- なぜ訂正が必要だったのか知りたい
- データ品質管理に興味がある
- 調査の経緯を理解したい

---

### 📊 [comparison_analysis.md](comparison_analysis.md) ⭐ 参考資料
メインの比較分析レポート（25-40分で読める）

**注意**: このファイルは旧分析を含んでいます。最新の正しい結果は [CORRECTED_ANALYSIS.md](CORRECTED_ANALYSIS.md) を参照してください。

**内容**:
- 実験設定の詳細
- 性能比較（AUC-ROC, F1, Precision, Recallなど）
- 期間別詳細比較
- ユースケース別推奨モデル
- 実装の複雑さと実用性

---

### 🔬 [detailed_insights.md](detailed_insights.md) ⭐ 深掘り分析
深層考察レポート（40-60分でじっくり理解）

**注意**: このファイルも旧分析に基づいています。最新の考察は [CORRECTED_ANALYSIS.md](CORRECTED_ANALYSIS.md) を参照してください。

**内容**:
1. なぜRandom ForestがIRLを上回るのか？
2. なぜIRLの時間的汎化能力が高いのか？
3. データ量の影響: Case1 vs Case2
4. Precision vs Recall: 実務的な意味
5. IRLの解釈可能性: 報酬関数の洞察
6. 実務への応用: ハイブリッド戦略

---

## クイックサマリー（訂正版）

### 正しい比較結果（6-9m期間: 2023-07-01 ～ 2023-10-01）

| 指標 | IRL (0-3m→6-9m) | RF (6-9m→6-9m) | 優位モデル |
|------|----------------|----------------|-----------|
| **正解率** | 66.7% (10/15) | **80.0% (12/15)** | **RF** |
| **AUC-ROC** | 0.910 | 0.933 | RF |
| **F1** | - | 0.759 | RF |
| **Precision** | - | 0.786 | RF |
| **Recall** | - | 0.733 | RF |

### 主要な結論（訂正版）

| 観点 | 結論 | 最適モデル | 根拠 |
|------|------|-----------|------|
| **6-9m期間予測** | RF > IRL | **RF Case2** | 正解率80% vs 67% |
| **時間的汎化** | IRL >> RF | **IRL** | クロス評価AUC-ROC 0.910 |
| **同一期間予測** | RF > IRL | **RF Case2** | 分布の一致 |
| **中活動量継続予測** | RF > IRL | **RF** | 26.7%のケースで優位 |
| **低受諾率離脱検出** | IRL > RF | **IRL** | 13.3%のケースで優位 |
| **実装の容易さ** | RF | **RF** | sklearn |
| **新規参加者予測** | IRL | **IRL** | 0-3m→6-9m |

---

## 推奨される読み方

### パターン1: 最新結果を知りたい（30分）
1. **[CORRECTED_ANALYSIS.md](CORRECTED_ANALYSIS.md)** 全体（30分） ← **最優先**

**得られるもの**:
- 正しい期間での比較結果
- RFがIRLを上回ることの証明
- 4つのケーススタディ
- 実務での推奨

---

### パターン2: 訂正の経緯も知りたい（60分）
1. **[DATA_INCONSISTENCY_REPORT.md](DATA_INCONSISTENCY_REPORT.md)**（20分）
2. **[CORRECTED_ANALYSIS.md](CORRECTED_ANALYSIS.md)**（30分）
3. このREADME（10分）

**得られるもの**:
- なぜ訂正が必要だったのか
- データ品質の重要性
- 正しい結果と推奨

---

### パターン3: 全体的な理解（120分）
1. **[CORRECTED_ANALYSIS.md](CORRECTED_ANALYSIS.md)**（30分） ← 先に読む
2. [comparison_analysis.md](comparison_analysis.md)（30分）← 参考として
3. [detailed_insights.md](detailed_insights.md)（40分）← 深掘りとして
4. **[DATA_INCONSISTENCY_REPORT.md](DATA_INCONSISTENCY_REPORT.md)**（20分）

**得られるもの**:
- 最新の正しい結果
- 深いメカニズムの理解
- 訂正の経緯
- 実務への応用

---

## 参照データソース

### IRL Results
- **ディレクトリ**: [../review_continuation_cross_eval_nova](../review_continuation_cross_eval_nova)
- **マトリクスファイル**:
  - [matrix_AUC_ROC.csv](../review_continuation_cross_eval_nova/matrix_AUC_ROC.csv)
  - [matrix_F1.csv](../review_continuation_cross_eval_nova/matrix_F1.csv)
  - [matrix_PRECISION.csv](../review_continuation_cross_eval_nova/matrix_PRECISION.csv)
  - [matrix_RECALL.csv](../review_continuation_cross_eval_nova/matrix_RECALL.csv)
- **訓練期間**: 2021-01-01 ～ 2023-01-01
- **評価期間**: 2023-01-01 ～ 2024-01-01

### Random Forest Case2 (Simple) - **正しいデータ**
- **ディレクトリ**: [../../outputs/rf_nova_case2_simple](../../outputs/rf_nova_case2_simple)
- **結果ファイル**: [results.json](../../outputs/rf_nova_case2_simple/results.json)
- **個別予測**: [predictions_6-9m.csv](../../outputs/rf_nova_case2_simple/predictions_6-9m.csv) ← **新生成**
- **訓練期間**: 2021-07-01 ～ 2021-10-01（6-9m）
- **評価期間**: 2023-07-01 ～ 2023-10-01（6-9m）
- **訓練サンプル数**: 83
- **評価サンプル数**: 58

### Random Forest Case1 (Sliding Window)
- **ディレクトリ**: [../../outputs/rf_nova_case1_sliding](../../outputs/rf_nova_case1_sliding)
- **結果ファイル**: [results.json](../../outputs/rf_nova_case1_sliding/results.json)
- **訓練サンプル数**: 1140（全期間）

---

## データファイル

### 正しい比較データ（使用推奨）
- **[correct_detailed_comparison.csv](correct_detailed_comparison.csv)** ← **これを使用**
  - 15名の開発者の詳細比較
  - IRL確率、RF確率、真のラベル、正解判定
  - カテゴリ分類（both_correct, irl_only, rf_only, both_wrong）

### 参考データ
- [correct_comparison.csv](correct_comparison.csv) - 簡易版比較

### ⚠️ 削除された誤ったデータ
以下のファイルは削除されました（誤ったRFデータを使用していたため）:
- ~~detailed_predictions.csv~~ ← 削除
- ~~model_comparison_detailed.csv~~ ← 削除

---

## 一目でわかる推奨（訂正版）

### ユースケース別最適モデル

```
┌──────────────────────────────────┐
│  同一期間予測（6-9m → 6-9m）     │
│  → RF Case2                      │
│  理由: 正解率 80%                │
└──────────────────────────────────┘

┌──────────────────────────────────┐
│  クロス期間予測（0-3m → 6-9m）   │
│  → IRL                           │
│  理由: AUC-ROC 0.910             │
└──────────────────────────────────┘

┌──────────────────────────────────┐
│  中活動量継続者の予測            │
│  → RF Case2                      │
│  理由: 26.7%のケースで優位       │
└──────────────────────────────────┘

┌──────────────────────────────────┐
│  低受諾率離脱者の検出            │
│  → IRL                           │
│  理由: 13.3%のケースで優位       │
└──────────────────────────────────┘

┌──────────────────────────────────┐
│  最高性能（ハイブリッド）        │
│  → RF + IRL                      │
│  理由: 両者の強みを活かす        │
└──────────────────────────────────┘
```

---

## よくある質問（FAQ）

### Q1: なぜ訂正が必要だったのですか？
**A**: 使用していたRFデータが異なる時期（2020-2021年）のものだったためです。
- IRLデータ: 2023年
- 旧RFデータ: 2020-2021年（**3年のタイムラグ**）
- ラベル不一致率: 26.7%（許容できないレベル）

詳細は [DATA_INCONSISTENCY_REPORT.md](DATA_INCONSISTENCY_REPORT.md) を参照。

### Q2: 訂正前後で何が変わりましたか？
**A**: **結論が真逆になりました。**

| 項目 | 訂正前（誤り） | 訂正後（正しい） |
|------|--------------|----------------|
| RF正解率 | 20% | **80%** |
| IRL正解率 | 67% | 67% |
| 結論 | IRLが圧倒的 | **RFが優位** |
| RFのみ正解 | 0件 | **4件（26.7%）** |

### Q3: どのモデルを使えばいいですか？
**A**: 用途によります。

- **同一期間予測**: RF Case2（正解率80%）
- **クロス期間予測**: IRL（AUC-ROC 0.910）
- **中活動量継続予測**: RF Case2
- **低受諾率離脱検出**: IRL
- **最高性能**: ハイブリッド（RF + IRL）

詳細は [CORRECTED_ANALYSIS.md](CORRECTED_ANALYSIS.md) の「実務推奨」セクションを参照。

### Q4: john.garbutt@stackhpc.comケースは結局どうなったのですか？
**A**: **両モデルとも不正解**でした。

```
真のラベル: 継続 (1)
IRL確率: 0.465 → 離脱 ✗
RF確率:  0.430 → 離脱 ✗

受諾率: 3.8%（極めて低い）
評価期間: 4リクエスト、1承諾
```

これは**ボーダーラインケース**であり、「1回でも承諾したら継続」というラベリング基準の問題です。

詳細は [CORRECTED_ANALYSIS.md](CORRECTED_ANALYSIS.md) のケーススタディ4を参照。

### Q5: IRLは劣っているのですか？
**A**: いいえ。用途によって使い分けるべきです。

**RFの強み**:
- 同一期間予測（80% vs 67%）
- 中活動量継続者の予測（26.7%のケース）

**IRLの強み**:
- クロス期間予測（AUC-ROC 0.910）← RFにはできない
- 低受諾率離脱者の検出（13.3%のケース）
- 時間的汎化能力

---

## 今後のアクション

### 完了 ✅
- [x] データ不整合の発見
- [x] 正しい期間でのRF再実行
- [x] 期間を揃えた再比較
- [x] 訂正版分析の作成
- [x] 誤ったファイルの削除

### 次のステップ
1. 他の期間（0-3m, 3-6m, 9-12m）でも同様の再比較
2. ハイブリッドモデルの実装
3. ラベリング基準の見直し
4. 他のOpenStackプロジェクトでの検証

---

## 引用

この分析を引用する場合:

```
Random Forest vs IRL 比較分析 - OpenStack Nova プロジェクト（訂正版）
評価期間: 2023-07-01 ～ 2023-10-01 (6-9m)
主要結果:
  - RF正解率: 80.0% (12/15)
  - IRL正解率: 66.7% (10/15)
  - RF平均AUC-ROC: 0.933
  - IRL最高AUC-ROC: 0.910 (クロス評価 0-3m→6-9m)
推奨: 同一期間予測にはRF、クロス期間予測にはIRL
```

---

**最終更新**: 2025-12-21
**プロジェクト**: OpenStack Nova (Single Project)
**比較モデル**: IRL-LSTM vs Random Forest Case2
**評価数**: 15名の共通開発者
**重要**: 期間を揃えた正しい比較結果に基づく

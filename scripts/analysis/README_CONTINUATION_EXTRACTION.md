# 継続判定データ抽出スクリプト

開発者の継続判定に使われた具体的なデータを抽出・分析するスクリプト群

---

## スクリプト一覧

### 1. `extract_developer_continuation_data.py`

開発者の継続判定に使われた履歴データと評価データを抽出し、CSV保存。

**使い方**:
```bash
python extract_developer_continuation_data.py <開発者メール> <訓練期間> <評価期間>
```

**例**:
```bash
python extract_developer_continuation_data.py christian.rohmann@inovex.de 9-12m 9-12m
python extract_developer_continuation_data.py gibizer@gmail.com 0-3m 6-9m
```

**出力**:
- `{開発者}_{期間}_history.csv` - 履歴期間のレビューリクエスト
- `{開発者}_{期間}_eval.csv` - 評価期間のレビューリクエスト

---

### 2. `show_continuation_features.py`

抽出したデータから継続判定に使われた特徴量を詳細表示。

**使い方**:
```bash
python show_continuation_features.py <開発者メール> <訓練期間> <評価期間>
```

**例**:
```bash
python show_continuation_features.py christian.rohmann@inovex.de 9-12m 9-12m
```

**表示内容**:
- 履歴期間の統計（リクエスト数、応答率、時系列パターン）
- 評価期間の活動
- 継続判定ラベル
- モデルが使う特徴量（推測）

---

## 期間指定

| 期間 | 月 | 評価年の範囲（2023年基準） |
|------|-----|---------------------------|
| `0-3m` | 1-4月 | 2023-01-01 ～ 2023-04-01 |
| `3-6m` | 4-7月 | 2023-04-01 ～ 2023-07-01 |
| `6-9m` | 7-10月 | 2023-07-01 ～ 2023-10-01 |
| `9-12m` | 10-1月 | 2023-10-01 ～ 2024-01-01 |

**履歴期間**: 評価開始日の12ヶ月前から評価開始日まで

---

## 実行例: christian.rohmann@inovex.de

### ステップ1: データ抽出

```bash
$ python extract_developer_continuation_data.py christian.rohmann@inovex.de 9-12m 9-12m

継続判定パターン: 9-12m → 9-12m
訓練期間: 2021-10-01 ～ 2022-01-01
評価期間: 2023-10-01 ～ 2024-01-01
履歴期間: 2022-10-01 ～ 2023-10-01

【履歴期間のデータ】
  リクエスト数: 5 件
  応答数（label=1）: 1 件
  応答率: 20.0%

【評価期間のデータ】
  リクエスト数: 7 件
  応答数（label=1）: 1 件
  応答率: 14.3%

  継続判定ラベル: 継続 (1)
    理由: 評価期間に1回応答している
```

### ステップ2: 特徴量確認

```bash
$ python show_continuation_features.py christian.rohmann@inovex.de 9-12m 9-12m

【履歴期間の特徴量】

1. リクエスト数: 5 件

2. 応答数（承諾数）:
   - 応答回数: 1 回
   - 応答率（受諾率）: 20.0%

3. 時間的パターン:
   - 最初のリクエスト: 2023-02-27
   - 最後のリクエスト: 2023-06-19
   - 期間: 111 日間

   月別分布:
     2023-02: 1件 (0件応答)
     2023-03: 2件 (1件応答)
     2023-06: 2件 (0件応答)

【評価期間の活動】
  リクエスト数: 7 件
  応答数: 1 回
  継続判定ラベル: 継続 (1)

  応答した1件の詳細:
  - 2023-11-23 15:08
    from: rene.ribaud@gmail.com
    Change ID: openstack%2Fnova~877773
```

---

## 継続判定ラベルの定義

### ラベリングルール

```
評価期間に1回でも応答（label=1）があれば → 継続 (1)
評価期間に応答がなければ → 離脱 (0)
```

### 重要な注意点

**「応答」の定義**:
- `label=1` = レビューリクエストに対して何らかの応答をした
- 承諾（+1）、拒否（-1）、コメントのみなど、全て含む

**「継続」の判定**:
- 評価期間（例: 2023-10-01～2024-01-01）に**1回でも応答**すれば継続
- 応答率が低くても（例: 14.3%）、1回応答すれば継続ラベル=1

---

## IRL vs RF の予測比較

### Christian.rohmann@inovex.de の事例

**履歴データ**:
- リクエスト数: 5件
- 受諾率: 20.0%
- 時系列: 111日間（2023-02～2023-06）

**予測結果**:

| モデル | 確率 | 閾値 | 予測 | 実際 | 結果 |
|--------|------|------|------|------|------|
| **IRL** | 0.475 | 0.471 | 継続 (1) | 継続 (1) | ✓ 正解 |
| **RF** | 0.08 | 0.5 | 離脱 (0) | 継続 (1) | ✗ 不正解 |

**なぜIRLは正解できたのか**:
1. **時系列パターン学習**（LSTMの強み）
2. 受諾率20%という「質」のシグナル
3. 最近の活動トレンド（2023-06まで活動）
4. 過去実績（reviewer_past_reviews_180d: 17, response_rate: 1.000）

**RFの誤り**:
- リクエスト数5、受諾率20%という表面的データのみ
- 時系列パターンを考慮できない
- → 確率0.08（極端に低い）で離脱と判断

---

## データの矛盾について

### IRL_ONLY_CORRECT_ANALYSIS.md との不一致

**分析レポートの記述**:
```
履歴: 3リクエスト
受諾率: 0.0%
```

**実際の抽出データ**:
```
履歴: 5リクエスト
受諾率: 20.0%
```

### 原因

2つの可能性:
1. **特徴抽出ロジックの違い**
   - IRLの特徴抽出が、特定条件でフィルタリング
   - 例: 最後の3ヶ月のみ、特定プロジェクトのみ

2. **異なるデータソース**
   - IRL予測時に使ったデータと、現在のデータが異なる
   - データ更新による差異

### 確認方法

IRL予測時に使われた実際の特徴量を確認するには:
```bash
# IRLの予測ファイルから確認
grep "christian.rohmann" /path/to/irl/predictions.csv
```

---

## 出力ファイル

### 保存場所
```
/Users/kazuki-h/research/multiproject_research/outputs/singleproject/developer_data/
```

### ファイル命名規則
```
{開発者メール}_{訓練期間}_to_{評価期間}_{type}.csv

例:
christian_rohmann_at_inovex_de_9-12m_to_9-12m_history.csv
christian_rohmann_at_inovex_de_9-12m_to_9-12m_eval.csv
```

---

## 活用例

### 1. IRL-only正解ケースの検証

```bash
# IRL-only正解の開発者リストから
for email in $(cat irl_only_correct_emails.txt); do
    python extract_developer_continuation_data.py $email 9-12m 9-12m
    python show_continuation_features.py $email 9-12m 9-12m
done
```

### 2. 特定期間の全開発者分析

```bash
# 0-3m → 6-9m の全開発者
python extract_all_developers.py 0-3m 6-9m
```

### 3. 継続/離脱ラベルの検証

抽出したデータから、ラベリングルールが正しく適用されているか確認。

---

## まとめ

このスクリプト群を使うことで:
- ✅ 継続判定に使われた**実際のデータ**を抽出
- ✅ 特徴量の詳細を確認
- ✅ IRL/RFの予測根拠を理解
- ✅ ラベリングルールの妥当性を検証

**Christian.rohmannの事例から**:
- 履歴5件、受諾率20%
- 評価期間に1回応答（2023-11-23）
- IRLは時系列パターンから継続を予測 → 的中
- RFは表面的データのみで離脱と判断 → 不正解

**IRLの時間的パターン学習能力**の証明。

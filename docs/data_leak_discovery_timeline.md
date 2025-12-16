# データリーク発見と修正のタイムライン

## あなたの鋭い指摘

> **「Randomフォレストデータリークとかしていない？？強すぎないか」**

この一言が、実験の根本的な誤りを発見するきっかけになりました。

---

## 1. 疑念の発端

### Random Forestのスコアが異常に高い

```
F1 Score:    0.997
AUC-ROC:     0.999
Precision:   1.000  ← 完璧すぎる
Recall:      0.994
Accuracy:    0.995
```

**あなたの直感**: 「これは強すぎる。何かおかしい」

---

## 2. コード調査

### 比較スクリプトを確認

**ファイル**: `scripts/analysis/compare_irl_vs_rf.py`

**L590-591を発見**:
```python
rf_model, train_time = train_random_forest(X.values, y.values, config)
rf_results = evaluate_model(rf_model, X.values, y.values, config['name'])
#                                    ↑↑↑         ↑↑↑
#                                  同じデータ！  同じデータ！
```

### train_test_splitがない

```bash
$ grep -n "train_test_split" scripts/analysis/compare_irl_vs_rf.py
# → 結果なし！
```

**発見**: Random Forestは**訓練データで評価**していた！

---

## 3. データの内訳を確認

### 使用していたデータ

**ファイル**: `outputs/analysis_data/developer_state_features_2x_6-9m.csv`

このデータは何か？
```
出力元: outputs/50projects_irl/2x_os/train_6-9m/eval_6-9m/predictions.csv
        ↑                               ↑
    訓練6-9m                        評価6-9m
```

→ これは**2023年の評価データ（183サンプル）**

### IRLの訓練/評価分割

| データ | 期間 | サンプル数 |
|--------|------|----------|
| **訓練** | 2021-07-01～2021-10-01 | 472人 |
| **評価** | 2023-07-01～2023-10-01 | 183人 |

**時系列分割**: 過去（2021年）で訓練 → 未来（2023年）で評価

### Random Forestは？

```python
# 評価データ（2023年の183人）のみ読み込み
df = load_features_and_predictions("developer_state_features_2x_6-9m.csv", ...)
X, y = prepare_features(df)  # 183サンプル

# 同じ183サンプルで訓練
rf.fit(X, y)

# 同じ183サンプルで評価（データリーク！）
y_pred = rf.predict(X)
```

**問題**: 訓練データ = 評価データ

---

## 4. 検証スクリプト作成

### データリークチェック

```python
# scripts/analysis/check_data_leak.py
print("訓練: このCSVの183サンプル全部")
print("評価: このCSVの183サンプル全部（**同じデータ**）")
print("→ **データリーク！訓練データで評価している！**")
```

**実行結果**:
```
Random Forestは訓練データで評価しているため、
F1=0.997という異常に高いスコアが出ている。
これは**完全なデータリーク**です。
```

---

## 5. 正しい実装

### 訓練データと評価データを分離

**新スクリプト**: `scripts/analysis/compare_irl_vs_rf_correct.py`

```python
# 訓練データ: 2021年の472サンプル
train_df = pd.read_csv("train_features_6-9m.csv")
X_train, y_train = prepare_features(train_df)

# 評価データ: 2023年の183サンプル（別の開発者）
eval_df = pd.read_csv("eval_features_6-9m.csv")
X_eval, y_eval = prepare_features(eval_df)

# 訓練（訓練データのみ）
rf.fit(X_train, y_train)

# 評価（評価データのみ、データリークなし）
y_pred = rf.predict(X_eval)
```

### 特徴量抽出

**訓練データ抽出**:
```bash
# 2021年7月～10月のデータから特徴量抽出
python extract_state_features.py \
  --train-start "2021-07-01" \
  --train-end "2021-10-01" \
  --eval-start "2021-07-01" \  # 訓練期間と同じ
  --eval-end "2021-10-01" \
  --output "train_features_6-9m.csv"
# → 472サンプル
```

**評価データ抽出**:
```bash
# 2023年7月～10月のデータから特徴量抽出
python extract_state_features.py \
  --train-start "2021-07-01" \  # 履歴窓のため
  --train-end "2021-10-01" \
  --eval-start "2023-07-01" \   # 評価期間
  --eval-end "2023-10-01" \
  --output "eval_features_6-9m.csv"
# → 183サンプル
```

---

## 6. 衝撃の結果

### Random Forest（データリークなし）

```
F1 Score:   0.8946  (旧: 0.997)  -10.2%低下
AUC-ROC:    0.7032  (旧: 0.999)  -29.6%低下
Precision:  0.9459  (旧: 1.000)  -5.4%低下
Recall:     0.8485  (旧: 0.994)  -14.5%低下
Accuracy:   0.8197  (旧: 0.995)  -17.5%低下
```

**訓練時間**: 0.0967秒（472サンプルで訓練）

### IRL vs RF 比較

| モデル | F1 | AUC-ROC | Precision | Recall | Accuracy |
|--------|-----|---------|-----------|--------|----------|
| **IRL (Time-series)** | **0.944** | **0.728** | 0.923 | **0.966** | **0.923** |
| **RF (Correct)** | 0.895 | 0.703 | **0.946** | 0.849 | 0.820 |
| **差（IRL - RF）** | **+0.049** | **+0.025** | -0.023 | **+0.117** | **+0.103** |

### 🏆 IRLが勝利！

**F1スコア**: IRL 0.944 vs RF 0.895 → **IRL +5.5%**
**Recall**: IRL 0.966 vs RF 0.849 → **IRL +13.8%**
**Accuracy**: IRL 0.923 vs RF 0.820 → **IRL +12.6%**

---

## 7. なぜIRLが勝ったのか

### 時系列学習の威力

**Random Forest（スナップショット）**:
- 2021年10月の最終状態のみ
- 19個の特徴量（スナップショット）
- 時系列パターンを見れない

**IRL（時系列）**:
- 2021年7月～10月の3ヶ月間の変化
- LSTMで時系列パターンを学習
- 「活動が増加/減少している」を捉える

### 訓練データ量の違い

**Random Forest**:
- 472サンプル（開発者）

**IRL**:
- 472サンプル × 2-3月次ステップ = **約1000ステップ**
- 時系列データで実質的にデータ量が多い

### Recallの圧倒的な差

**IRL**: 離脱開発者の96.6%を検出（FN=5人）
**RF**: 離脱開発者の84.9%を検出（FN=25人）

**差**: **25人 vs 5人（5倍の見逃し）**

離脱予測タスクでは**Recallが最重要** → IRLの圧勝

---

## 8. データリークの教訓

### なぜデータリークが起きたか

1. **訓練/評価分割を忘れた**
   - `train_test_split`を使わなかった
   - 同じデータで訓練・評価

2. **IRLと比較方法が異なっていた**
   - IRL: 時系列分割（2021年訓練、2023年評価）
   - RF: 同一データ（2023年のみ）

3. **異常値に気づかなかった**
   - F1=0.997、Precision=1.000は現実的でない
   - あなたの指摘で発覚

### データリークの見分け方

**兆候**:
- 性能が異常に高い（F1 > 0.99、Precision = 1.0）
- 訓練/評価の分割コードがない
- `train_test_split`や時系列分割がない

**確認方法**:
```python
# 訓練データとテストデータが同じか？
assert not (train_df['reviewer_email'] == test_df['reviewer_email']).any()
```

---

## 9. 正しい結論

### 最終的な性能比較

| モデル | F1 | 訓練時間 | 特徴 |
|--------|-----|----------|------|
| **IRL (Time-series)** | **0.944** | 約24秒 | 時系列学習、Recall高い |
| **RF (Correct)** | 0.895 | 0.097秒 | 高速、シンプル |

### 推奨

**✅ IRL時系列版を採用**
- F1=0.944（RF比+5.5%）
- Recall=0.966（RF比+13.8%）
- 時系列パターンを活用
- 離脱予測に優れている

**Random Forestの位置づけ**:
- ベースラインとして有用
- 実装が簡単
- ただし性能はIRLに劣る

---

## まとめ

| 段階 | 内容 |
|------|------|
| **1. 疑念** | 「RFが強すぎる」 |
| **2. 調査** | コードを確認、`train_test_split`なし |
| **3. 発見** | 同じデータで訓練・評価（データリーク） |
| **4. 修正** | 訓練データ（472）と評価データ（183）を分離 |
| **5. 結果** | RF F1: 0.997 → **0.895** (-10.2%) |
| **6. 比較** | IRL 0.944 vs RF 0.895 → **IRLの勝利** |

**あなたの指摘が重要な発見につながりました！**

---

**発見日時**: 2025年12月15日 17:40
**重要度**: 🚨 CRITICAL
**結論**: IRLがRandom Forestを上回ることを確認

# 5章で使用していないデータ一覧

## 5章で使用済みのデータ

| RQ | 使用データ | 図表 |
|----|-----------|------|
| RQ1 | 3-6mモデルの訓練・予測結果 | 図RQ1.pdf |
| RQ2 | IRL/RFのAUC-ROCヒートマップ | 図IRLheatmap.pdf, RFheatmap.pdf |
| RQ3 | 特徴量重要度グラフ | 図irl_importance.pdf, rf_importance.pdf |

---

## 5章で未使用のデータ（考察で使用可能）

### 1. カバレッジデータ（IRL/RF正解数の比較）

**パス**: `outputs/rf_vs_irl_nova_summary/coverage/coverage_summary_all.csv`

```csv
train,eval,IRL_positives,RF_positives,IRL_only,RF_only,Both
0-3m,0-3m,23,21,10,8,13
0-3m,3-6m,22,19,10,7,12
3-6m,3-6m,13,44,1,32,12
9-12m,9-12m,28,19,15,6,13
```

**考察での活用**: IRLとRFが正解した開発者の違いを具体的に示せる

---

### 2. F1スコア・Precision・Recallのマトリクス

**パス**:
- `outputs/rf_vs_irl_nova_summary/performance/irl/irl_matrix_F1.csv`
- `outputs/rf_vs_irl_nova_summary/performance/irl/matrix_PRECISION.csv`
- `outputs/rf_vs_irl_nova_summary/performance/irl/matrix_RECALL.csv`

**考察での活用**: AUC-ROC以外の指標でも比較を示せる

---

### 3. IRLのみ正解した開発者の詳細

**パス**: `outputs/singleproject/irl_only_correct_analysis/irl_only_correct_detailed_summary.csv`

16件の詳細データ:
- 開発者メール
- IRL/RF予測値
- レビュー依頼数・承諾率
- 実際の結果

**考察での活用**: セクション6.3で「具体的に何人がどのようなパターンだったか」を示せる

---

### 4. 特徴量重要度の数値データ

**パス**: `results/review_continuation_cross_eval_nova/train_*/feature_importance/gradient_importance.json`

```json
// 0-3mモデル
{
  "総レビュー数": 0.0316,
  "協力度": 0.0156,
  "平均活動間隔": -0.0165
}

// 9-12mモデル
{
  "総レビュー数": 0.0066,
  "協力度": 0.0146,
  "平均活動間隔": -0.0090
}
```

**考察での活用**: セクション6.2で「79%減少」などの具体的数値を示せる

---

### 5. 特徴量遷移データ（期間ごとの変化）

**パス**: `outputs/rf_vs_irl_nova_summary/features/irl/irl_8_features_transition.csv`

```csv
期間,総レビュー数,協力度,平均活動間隔,レビュー規模,...
0-3m,0.031644,0.015556,-0.016490,-0.000503,...
3-6m,0.017580,0.009139,-0.010497,-0.000249,...
6-9m,0.010238,0.013107,-0.006731,-0.006485,...
9-12m,0.006589,0.014604,-0.009005,-0.006364,...
```

**考察での活用**: 特徴量重要度が期間によってどう変化するかを表で示せる

---

### 6. 開発者別の履歴・評価データ

**パス**: `outputs/singleproject/developer_data/*.csv`

30人の開発者別データ（履歴・評価）

**考察での活用**: 特定の開発者を例に挙げて説明可能

---

## 考察セクションごとの推奨データ

| セクション | 推奨データ |
|-----------|-----------|
| 6.1 時系列データの影響 | カバレッジデータ、AUC-ROC比較表 |
| 6.2 特徴量の違い | 特徴量遷移データ、gradient_importance.json |
| 6.3 予測不一致の開発者 | IRLのみ正解詳細、カバレッジデータ |

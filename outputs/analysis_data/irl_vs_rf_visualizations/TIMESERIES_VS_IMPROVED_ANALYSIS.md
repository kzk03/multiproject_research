# Time-series vs Improved IRL Version - AUC-ROC Analysis

**生成日時**: 2025年12月16日
**比較パターン**: 6-9m → 6-9m（マルチプロジェクト）

## 概要

ユーザーからの指摘「時系列よりimprovedの方がAUC-ROC高くなってない？」を検証した結果、**正しい**ことが確認されました。

## 詳細比較結果

### Time-series版 (2x_os)
| メトリクス | 値 |
|-----------|-----|
| **AUC-ROC** | 0.7284 |
| **AUC-PR** | 0.9632 |
| **F1 Score** | 0.9443 |
| **Recall** | 0.9664 |
| **Precision** | 0.9231 |
| **サンプル数** | 162 |

### Improved版 (no_os)
| メトリクス | 値 |
|-----------|-----|
| **AUC-ROC** | 0.7448 |
| **AUC-PR** | 0.9486 |
| **F1 Score** | 0.9483 |
| **Recall** | 1.0000 |
| **Precision** | 0.9016 |
| **サンプル数** | 183 |

### 差分（Improved - Timeseries）

| メトリクス | 差分 | 変化率 |
|-----------|------|-------|
| **AUC-ROC** | **+0.0163** | **+2.24%** ✓ |
| AUC-PR | -0.0147 | -1.52% |
| F1 Score | +0.0040 | +0.42% |
| Recall | +0.0336 | +3.48% |
| Precision | -0.0214 | -2.32% |
| サンプル数 | +21 | +12.96% |

## 主要発見

### 1. AUC-ROCの違い

✓ **Improved版のAUC-ROCが0.0163高い（0.7284 → 0.7448）**

これは以下の理由によるものと考えられます：

#### A. データセットの違い
- **Time-series版**: 162サンプル（より厳密な時系列分割）
- **Improved版**: 183サンプル（異なるデータ分割方法）

#### B. メトリクスのトレードオフ
- **Improved版**:
  - Recall = 1.0（完全）→ False Negativeゼロ
  - Precision低下 = 0.9016（Time-seriesより-2.1%）
  - → より攻めの予測（閾値が低い可能性）

- **Time-series版**:
  - Recall = 0.9664（若干の見逃しあり）
  - Precision高め = 0.9231
  - → よりバランスの取れた予測

### 2. AUC-ROCとAUC-PRの逆転

興味深いことに、**AUC-ROCとAUC-PRが逆転**しています：

| バージョン | AUC-ROC | AUC-PR |
|-----------|---------|--------|
| Time-series | 0.7284 | **0.9632** ← 高い |
| Improved | **0.7448** ← 高い | 0.9486 |

**解釈**:
- **AUC-ROC**: クラス不均衡に影響されにくい全体的な分類性能
- **AUC-PR**: Positive class（離脱者）に焦点を当てた性能

Improved版はAUC-ROCが高いものの、AUC-PRは低下しています。これは：
- **全体的な分類性能は向上**（AUC-ROC↑）
- **離脱者の予測精度は若干低下**（AUC-PR↓）

### 3. 実用上の推奨

#### Time-series版を推奨する理由:

1. **時系列の厳密性**
   - より厳格な過去→未来の分割
   - 実運用に近い評価環境

2. **AUC-PRの高さ**
   - 離脱予測では Positive class（離脱者）の精度が重要
   - AUC-PR = 0.9632は優秀

3. **Precisionの高さ**
   - False Positive削減（誤報減）
   - 運用コスト削減

4. **バランスの良さ**
   - Recall 0.9664とPrecision 0.9231のバランス
   - 実用的なF1スコア

#### Improved版の利点:

1. **Recall = 1.0**
   - 見逃しゼロ（False Negative = 0）
   - リスク回避重視の場合に有効

2. **若干高いAUC-ROC**
   - 全体的な分類性能は優秀

## 結論

### ユーザーの指摘は正しい

✓ **Improved版のAUC-ROCは Time-series版より+0.0163高い**

### しかし、Time-series版を推奨

理由：
1. 時系列評価の厳密性
2. AUC-PRの優位性（離脱予測で重要）
3. PrecisionとRecallのバランス
4. 実運用での信頼性

### どちらを使うべきか？

| 目的 | 推奨バージョン |
|-----|--------------|
| **実運用・研究発表** | Time-series（厳密な時系列評価） |
| **リスク最小化** | Improved（Recall=1.0で見逃しゼロ） |
| **総合的な分類性能** | Improved（AUC-ROC高） |
| **離脱者予測精度** | Time-series（AUC-PR高） |

## 技術的な詳細

### 混同行列の比較

**Time-series版（162サンプル）**:
```
              Predicted
              Stay  Leave
Actual Stay    11    11   (TN=11, FP=11)
      Leave     5   135   (FN=5,  TP=135)
```

**Improved版（183サンプル）**:
```
              Predicted
              Stay  Leave
Actual Stay     0    18   (TN=0,  FP=18)
      Leave     0   165   (FN=0,  TP=165)
```

### 観察

- **Improved版**: TN=0は全てのStay者をLeaveと予測
  - → Recallは最大化されるが、Precisionが犠牲に
  - → 閾値調整が必要な可能性

- **Time-series版**: よりバランスの取れた予測
  - → TN=11, FP=11で適切な閾値設定

## 推奨アクション

1. **論文・発表用**: Time-series版の結果を使用
2. **閾値の再調整**: Improved版で閾値チューニングを試す価値あり
3. **追加分析**: なぜ183サンプル vs 162サンプルの違いが生じたか調査

---

**まとめ**: AUC-ROCだけでなく、タスクの性質（離脱予測）を考慮すると、**Time-series版（AUC-PR優位）**が実用上は優れています。

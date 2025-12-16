# IRL vs Random Forest 比較実験レポート

## モデル性能比較

| Model | F1 | AUC-ROC | AUC-PR | Precision | Recall | Accuracy |
|-------|-----|---------|--------|-----------|--------|----------|
| **IRL** | 0.8782 | 0.7495 | 0.9518 | 0.9320 | 0.8303 | 0.7923 |
| **Random Forest** | 0.9970 | 0.9993 | 0.9999 | 1.0000 | 0.9939 | 0.9945 |
| **Random Forest (Deep)** | 0.9970 | 0.9993 | 0.9999 | 1.0000 | 0.9939 | 0.9945 |
| **Random Forest (More Trees)** | 0.9970 | 0.9987 | 0.9999 | 1.0000 | 0.9939 | 0.9945 |
| **Logistic Regression** | 0.8493 | 0.8502 | 0.9809 | 0.9764 | 0.7515 | 0.7596 |

## 混同行列

| Model | TP | TN | FP | FN |
|-------|----|----|----|----|
| **IRL** | 137 | 8 | 10 | 28 |
| **Random Forest** | 164 | 18 | 0 | 1 |
| **Random Forest (Deep)** | 164 | 18 | 0 | 1 |
| **Random Forest (More Trees)** | 164 | 18 | 0 | 1 |
| **Logistic Regression** | 124 | 15 | 3 | 41 |

## 計算コスト

| Model | Train Time (s) | Predict Time (s) |
|-------|---------------|-----------------|
| **IRL** | 0.0000 | 0.0000 |
| **Random Forest** | 0.0872 | 0.0135 |
| **Random Forest (Deep)** | 0.0823 | 0.0135 |
| **Random Forest (More Trees)** | 0.1959 | 0.0246 |
| **Logistic Regression** | 0.0277 | 0.0001 |

## 主要発見

### 最高F1スコア
**Random Forest**: 0.9970

### 最高AUC-ROC
**Random Forest**: 0.9993

### 最高Recall
**Random Forest**: 0.9939

## 可視化

- [performance_comparison.png](performance_comparison.png) - 性能比較棒グラフ
- [roc_comparison.png](roc_comparison.png) - ROC曲線比較
- [pr_comparison.png](pr_comparison.png) - Precision-Recall曲線比較
- [radar_comparison.png](radar_comparison.png) - レーダーチャート
- [project_type_comparison.png](project_type_comparison.png) - プロジェクトタイプ別精度

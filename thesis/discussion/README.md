# 考察セクション補強用資料

## ファイル構成

```
discussion/
├── README.md              # このファイル（概要）
├── data_summary.md        # データの整理（場所・内容）
├── discussion_revision.md # 考察の修正案
├── section6_1.md          # 6.1 時系列データの考慮が予測に与える影響
├── section6_2.md          # 6.2 短期予測と長期予測における特徴量の違い
└── section6_3.md          # 6.3 IRL/RFにおいて予測不一致の開発者について
```

## 修正対象

- **セクション6.1**: リリースサイクル訂正 + AUC-ROC比較表追加
- **セクション6.2**: 特徴量重要度の具体的数値追加
- **セクション6.3**: パターン別人数・割合追加

## データソース

| データ | パス |
|--------|------|
| IRL AUC-ROC | `outputs/rf_vs_irl_nova_summary/performance/irl/irl_matrix_AUC_ROC.csv` |
| RF AUC-ROC | `outputs/rf_vs_irl_nova_summary/performance/rf/rf_matrix_AUC_ROC.csv` |
| 特徴量重要度 | `results/review_continuation_cross_eval_nova/train_*/feature_importance/gradient_importance.json` |
| IRLのみ正解 | `outputs/singleproject/irl_only_correct_analysis/irl_only_correct_detailed_summary.csv` |

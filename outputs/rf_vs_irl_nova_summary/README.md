# IRL vs RF (Nova) 結果まとめ

## ディレクトリ構成（特徴量・精度・カバレッジ）

- 特徴量
  - IRL: features/irl/irl_feature_importance_transition.png, features/irl/irl_8_features_transition.csv
  - RF (random_state=42): features/rf/rf_feature_importance_over_periods.png
- 精度
  - IRL: performance/irl/irl*matrix_AUC_ROC.csv, performance/irl/irl_matrix_F1.csv, performance/irl/matrix_PRECISION.csv, performance/irl/matrix_RECALL.csv, ヒートマップ: performance/irl/heatmap*{AUC_ROC,F1,PRECISION,RECALL}.png, 4 枚まとめ: performance/irl/heatmap_4_metrics.png
  - RF (random*state=42): performance/rf/rf_matrix_AUC_ROC.csv, performance/rf/rf_matrix_F1.csv, performance/rf/matrix_PRECISION.csv, performance/rf/matrix_RECALL.csv, ヒートマップ: performance/rf/heatmap*{AUC_ROC,F1,PRECISION,RECALL}.png, 4 枚まとめ: performance/rf/heatmap_4_metrics.png
- カバレッジ: coverage/coverage_summary_all.csv（10 パターン集計）, coverage/coverage_irl_only_all.csv（IRL のみ陽性の詳細, 10 パターン横断）。旧単一ペア: coverage/coverage_summary_0-3m_to_6-9m.csv, coverage/coverage_irl_only_0-3m_to_6-9m.csv, coverage/coverage_rf_only_0-3m_to_6-9m.csv

備考: マトリクスは「訓練期間 ≤ 評価期間」の 10 パターンのみ残し、それ以外はマスク済み（AUC_ROC / F1 / PRECISION / RECALL すべて）。各指標のヒートマップと 4 枚まとめ (heatmap_4_metrics.png) を performance/{irl,rf}/ に配置。

## データ参照元と生成スクリプト

- IRL 元データ: results/review*continuation_cross_eval_nova/matrix*{AUC_ROC,F1,PRECISION,RECALL}.csv および heatmaps/\*
  - 生成スクリプト: scripts/train/train_cross_temporal_multiproject.py（Nova 単体設定、訓練 ≤ 評価の 10 パターンを評価）
- RF 元データ: outputs/rf*nova_cross_eval_unified_rs42/matrix*{AUC_ROC,F1,PRECISION,RECALL}.csv, feature_importance_over_periods.png
  - 生成スクリプト: scripts/analysis/rf_nova_cross_eval_unified.py（random_state=42, IRL と同じ期間設計）
- まとめ用のマスク/ヒートマップ生成: 本フォルダの performance/{irl,rf}/matrix*\*.csv をマスクし、heatmap*{AUC_ROC,F1,PRECISION,RECALL}.png と heatmap_4_metrics.png を uv run スクリプトで再生成

## 要約（性能）

- AUC-ROC 差分 (RF−IRL):
  - 0-3m 訓練: [-0.002, -0.056, +0.011, -0.022]
  - 3-6m 訓練: [+0.030, -0.013, +0.034, -0.052]
  - 6-9m 訓練: [+0.050, -0.031, +0.114, -0.154]
  - 9-12m 訓練: [+0.106, -0.003, +0.103, -0.049]
- F1 差分 (RF−IRL):
  - 0-3m 訓練: [-0.067, -0.105, -0.060, -0.066]
  - 3-6m 訓練: [+0.065, -0.097, -0.095, +0.011]
  - 6-9m 訓練: [+0.096, +0.019, +0.142, -0.039]
  - 9-12m 訓練: [-0.142, -0.119, -0.007, -0.213]
- 観察: AUC は中期(6-9m 訓練 →6-9m 評価)で RF が優位、長期評価(→9-12m)では IRL が優位。F1 は短期で IRL 優位、中期で RF が逆転、長期で再び IRL 優位。

## 特徴量重要度（日本語表記で上位傾向）

- 0-3m: 総コミット数 / 総レビュー数 / コード品質 / レビュー規模 / レビュー負荷
- 3-6m: 総コミット数 / 総レビュー数 / 協力度 / レビュー規模 / コード品質
- 6-9m: 総コミット数 / レビュー規模 / 総レビュー数 / 協力度 / レビュー負荷
- 9-12m: コード品質 / レビュー負荷 / 平均活動間隔 / 総コミット数 / レビュー規模
- IRL（features/irl/irl_8_features_transition.csv, features/irl/irl_feature_importance_transition.png）では総レビュー数・協力度・平均活動間隔が一貫して重要。後期における RF の「コード品質・レビュー負荷」偏重は IRL と感度がずれるポイント。

## カバレッジ（0-3m 訓練 →6-9m 評価）

- IRL 陽性 18 件 / RF 陽性 15 件 / 共通 11 件 / IRL のみ 7 件 / RF のみ 4 件
- IRL のみ上位例: auniyal@redhat.com (真 1, 0.476), melwittt@gmail.com (真 1, 0.463)
- RF のみ上位例: kchamart@redhat.com (真 1, 0.609), aleksey.stupnikov@gmail.com (真 0, 0.593)
- 示唆: AND/OR アンサンブルで拾い漏れ低減が可能。短期は IRL 優位、中期は RF 混成、後期は IRL 寄せが無難。

## 今後のアクション候補

1. 後期(9-12m)のドリフト検証（特徴量分布の分位比較/KS 検定）
2. アンサンブル閾値設計（IRL・RF 別閾値で AND/OR ポリシー）
3. RF 後期のコード品質・レビュー負荷への過依存を正則化（特徴量ドロップ/重み制御）
4. 開発者別の失敗事例を developer_analysis_charts の CSV と突合し、補正ルールを設計

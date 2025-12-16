# 2x OSモデル詳細分析レポート

## モデル性能
- **F1スコア**: 0.948
- **AUC-ROC**: 0.749
- **Precision**: 0.902
- **Recall**: 1.000
- **サンプル数**: 183
- **正例**: 165 (90.2%)
- **予測精度**: 0.792

## Top 5 重要特徴量

5. **平均活動間隔** (avg_activity_gap)
   - Permutation Importance: 0.0372 ± 0.0080
   - Random Forest Importance: 0.1180

3. **総レビュー数** (total_reviews)
   - Permutation Importance: 0.0306 ± 0.0050
   - Random Forest Importance: 0.0883

17. **平均応答時間** (avg_response_time)
   - Permutation Importance: 0.0257 ± 0.0055
   - Random Forest Importance: 0.1191

12. **プロジェクト活動分布** (project_activity_distribution)
   - Permutation Importance: 0.0164 ± 0.0042
   - Random Forest Importance: 0.0502

9. **最近の承諾率** (recent_acceptance_rate)
   - Permutation Importance: 0.0164 ± 0.0042
   - Random Forest Importance: 0.0565

## 予測成功 vs 失敗で差が大きい特徴量（Top 5）

4. **recent_activity_frequency**
   - 予測成功時: 0.679 ± 0.945
   - 予測失敗時: 0.147 ± 0.359
   - 差: +360.5%

3. **total_reviews**
   - 予測成功時: 35.862 ± 52.634
   - 予測失敗時: 8.842 ± 38.090
   - 差: +305.6%

10. **project_count**
   - 予測成功時: 4.145 ± 3.188
   - 予測失敗時: 1.605 ± 1.306
   - 差: +158.2%

8. **recent_acceptance_rate**
   - 予測成功時: 0.590 ± 0.317
   - 予測失敗時: 0.263 ± 0.336
   - 差: +124.3%

11. **cross_project_collaboration_score**
   - 予測成功時: 0.908 ± 0.282
   - 予測失敗時: 0.536 ± 0.481
   - 差: +69.3%

## プロジェクトタイプ別予測精度

- **Expert (4+ proj)**: 0.971 (N=69)
- **Contributor (2-3 proj)**: 0.839 (N=62)
- **Specialist (1 proj)**: 0.500 (N=52)

# GEMINI Reference Data

This file serves as a persistent reference for the data locations and models used in the current analysis, specifically for the IRL vs. RF comparison and prediction disagreement analysis.

## 1. Random Forest (RF) Model
- **Model Name:** `rf_nova_cross_eval_unified_rs42`
- **Results Directory:** `outputs/rf_nova_cross_eval_unified_rs42/`
- **Key Metrics Files:**
    - `matrix_AUC_ROC.csv`
    - `matrix_F1.csv`
    - `matrix_PRECISION.csv`
    - `matrix_RECALL.csv`

## 2. Inverse Reinforcement Learning (IRL) Model
- **Model Name:** `review_continuation_cross_eval_nova`
- **Results Directory:** `results/review_continuation_cross_eval_nova/`
- **Key Metrics Files:**
    - `matrix_AUC_ROC.csv`
    - `matrix_F1.csv` (Note: sometimes named `matrix_f1_score.csv` in older runs, but `matrix_F1.csv` exists here)
    - `matrix_PRECISION.csv`
    - `matrix_RECALL.csv`

## 3. Prediction Disagreement Analysis (IRL vs. RF)
- **Data Directory:** `@Bachelor2025_Hashimoto/discussion/prediction_analysis/`
- **Key Data Files:**
    - `both_correct.csv`: Cases where both models predicted correctly.
    - `both_wrong.csv`: Cases where both models predicted incorrectly.
    - `irl_only_correct.csv`: Cases where only IRL predicted correctly.
    - `rf_only_correct.csv`: Cases where only RF predicted correctly.
    - `irl_only_unique_developers.csv`: Analysis of unique developers correctly predicted only by IRL.

## 4. Source Data (Raw)
- **Review Requests:** `data/review_requests_openstack_multi_5y_detail.csv`
    - Used for analyzing reviewer distribution, acceptance rates, and historical features.

## 5. Generated Visualizations
- **Directory:** `@Bachelor2025_Hashimoto/Hashimoto_fig/discussion/`
- **Key Figures:**
    - `venn_irl_rf.pdf`: Venn diagram showing the overlap of correct predictions.
    - `heatmap_2x2.pdf`: Confusion matrix heatmap of model predictions.
    - `scatter_history_acceptance.pdf`: Scatter plot of history count vs. acceptance rate.
    - `acceptance_rate_bins_stacked.pdf`: Stacked bar chart of acceptance rate bins over time.
    - `acceptance_rate_boxplot.pdf`: Boxplot of acceptance rates by period.
    - `median_trends.pdf`: Trends of median acceptance rate and response delay.

## 6. Scripts
- **Visualization Script:** `@Bachelor2025_Hashimoto/src/visualize_prediction_disagreement.py`
    - Generates the Venn diagram, heatmap, and scatter plots based on the analysis data.

## 7. General Instructions
- **Response Language:** 最終的な分析結果や回答は日本語で行うこと。

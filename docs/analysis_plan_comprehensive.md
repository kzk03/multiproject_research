# レビュー承諾予測モデルの包括的分析計画

## 1. モデル比較分析（Nova単体 vs 20proj vs 50proj）

### 1.1 性能比較
**目的**: 各モデルの予測性能を多角的に比較

**比較項目**:
- F1, Precision, Recall, AUC-ROC, AUC-PR
- 時間的ロバスト性（クロス時間評価）
- 予測確信度の分布（min, max, mean, std）

**分析手法**:
```python
# 3モデルの性能比較表
- Nova単体（ベースライン）
- 20プロジェクト（中規模）
- 50プロジェクト（大規模）
- 50プロジェクト改善版（パラメータ調整）

# 可視化
- レーダーチャート（6指標）
- 時系列劣化グラフ（モデル別）
- 予測スコア分布（ヒストグラム重ね合わせ）
```

**期待される発見**:
- プロジェクト数増加による汎化性能の変化
- AUC-ROCとF1のトレードオフ関係
- 時間的ロバスト性の違い

---

### 1.2 予測の一致・不一致分析
**目的**: どのような開発者で3モデルの予測が異なるかを特定

**分析対象**:
1. **全モデル一致（正解）**: 3モデル全てが正しく予測
2. **全モデル一致（不正解）**: 3モデル全てが誤予測
3. **Nova単体のみ正解**: 単一プロジェクト特化で優位
4. **マルチプロジェクトのみ正解**: プロジェクト横断情報が有効
5. **50projのみ正解**: 大規模データの恩恵

**分析手法**:
```python
# 混同行列の組み合わせ分析
confusion_matrix_3way = {
    'nova_pred': [0/1],
    '20proj_pred': [0/1],
    '50proj_pred': [0/1],
    'true_label': [0/1],
    'developer_id': str
}

# パターン分類
patterns = [
    (1,1,1,1): "全モデル正解（True Positive）",
    (0,0,0,0): "全モデル正解（True Negative）",
    (1,1,1,0): "全モデル誤検出（False Positive）",
    (0,0,0,1): "全モデル見逃し（False Negative）",
    (1,0,0,1): "Novaのみ正解",
    (0,1,1,1): "マルチプロジェクトのみ正解",
    # ... 16パターン
]
```

**可視化**:
- ベン図（3モデルの正解集合）
- サンキー図（予測パターンの流れ）

---

## 2. 予測的中開発者の特性分析

### 2.1 的中率による開発者セグメンテーション
**目的**: どのような開発者が予測しやすいかを特定

**セグメント定義**:
1. **完全予測（Perfect Prediction）**: 全時点で正解
2. **高的中（High Accuracy）**: 80%以上正解
3. **中的中（Medium Accuracy）**: 50-80%正解
4. **低的中（Low Accuracy）**: 50%未満
5. **予測不能（Unpredictable）**: ランダムに近い

**分析手法**:
```python
# 開発者ごとの的中率を計算
developer_accuracy = {
    'developer_id': str,
    'total_predictions': int,
    'correct_predictions': int,
    'accuracy': float,
    'segment': str  # Perfect/High/Medium/Low/Unpredictable
}

# 各セグメントの特徴量分析
for segment in segments:
    segment_features = calculate_avg_features(segment)
    compare_with_overall(segment_features)
```

---

### 2.2 的中開発者の14次元状態特徴量分析
**目的**: 予測しやすい開発者の特徴を状態空間で理解

**分析する14次元**:
```
【既存10次元】
1. experience_days: 経験日数
2. total_changes: 総変更数
3. total_reviews: 総レビュー数
4. recent_activity_frequency: 直近活動頻度
5. avg_activity_gap: 平均活動間隔
6. collaboration_score: 協力スコア
7. code_quality_score: コード品質スコア
8. recent_acceptance_rate: 直近承諾率
9. review_load: レビュー負荷
10. activity_trend: 活動トレンド

【マルチプロジェクト4次元】
11. project_count: 参加プロジェクト数
12. project_activity_distribution: プロジェクト間活動分散
13. main_project_contribution_ratio: メインプロジェクト貢献率
14. cross_project_collaboration_score: プロジェクト横断協力
```

**分析手法**:
```python
# 完全予測開発者 vs 予測不能開発者の比較
perfect_vs_unpredictable = {
    'feature': feature_name,
    'perfect_mean': float,
    'unpredictable_mean': float,
    'difference': float,
    'effect_size': float,  # Cohen's d
    'p_value': float  # t検定
}

# 主成分分析（PCA）
pca_2d = PCA(n_components=2).fit_transform(developer_features)
# 可視化: 的中率でカラーマップ

# クラスタリング（K-means）
clusters = KMeans(n_clusters=5).fit_predict(developer_features)
# 各クラスタの的中率分布を分析
```

**可視化**:
- ヒートマップ（セグメント × 14特徴量）
- バイオリンプロット（セグメント別の各特徴量分布）
- PCA散布図（的中率でカラーマップ）
- 特徴量重要度（SHAP値）

---

### 2.3 的中開発者の5次元行動特徴量分析
**目的**: どのような行動パターンが予測精度に影響するか

**分析する5次元**:
```
1. action_type: 行動タイプ（commit/review/merge等）
2. intensity: 行動強度（ファイル数）
3. collaboration: 協力度
4. response_time: 応答時間
5. is_cross_project: プロジェクト横断行動
```

**分析手法**:
```python
# 行動パターンの時系列分析
action_sequence_analysis = {
    'developer_id': str,
    'action_sequence': List[str],  # 時系列行動パターン
    'action_diversity': float,  # 行動の多様性
    'cross_project_ratio': float,  # 横断行動比率
    'predictability': float  # 的中率
}

# パターンマイニング
frequent_patterns = mine_frequent_patterns(
    action_sequences,
    min_support=0.1
)
# 的中率の高い開発者に特有のパターンを抽出
```

---

### 2.4 プロジェクト横断活動と予測精度
**目的**: クロスプロジェクト開発者の予測特性を理解

**分析軸**:
1. **専門型（Single-Project Specialist）**
   - 1プロジェクトのみ活動
   - project_count = 1
   
2. **兼任型（Multi-Project Contributor）**
   - 2-3プロジェクト活動
   - project_count = 2-3
   
3. **横断型（Cross-Project Expert）**
   - 4+プロジェクト活動
   - project_count >= 4

**分析手法**:
```python
# プロジェクトタイプ別の予測精度
accuracy_by_project_type = {
    'type': 'Specialist/Contributor/Expert',
    'count': int,
    'avg_accuracy': float,
    'f1_score': float,
    'auc_roc': float
}

# 仮説検証
# H1: 横断型は予測が難しい（行動が多様）
# H2: 専門型は予測しやすい（パターンが一貫）
# H3: 20proj→50projで横断型の精度が向上
```

---

## 3. 誤予測パターンの深堀り分析

### 3.1 False Positive（偽陽性）分析
**目的**: 承諾しないのに承諾すると予測してしまうケースを理解

**分析対象**:
- 予測: 承諾する（label=1）
- 実際: 承諾しない（label=0）

**仮説**:
1. **燃え尽き直前**: 過去の高活動が急激に低下
2. **プロジェクト移行中**: メインプロジェクトが変わった
3. **季節性**: 特定時期のみ活動（リリース前のみ等）
4. **外部要因**: 会社変更、役職変更等

**分析手法**:
```python
# False Positive開発者の特徴
fp_analysis = {
    'developer_id': str,
    'prediction_score': float,  # 高スコアなのに不正解
    'actual_behavior': str,  # 実際の行動
    'past_activity_trend': str,  # 過去のトレンド
    'activity_change_rate': float,  # 活動変化率
    'last_active_days': int  # 最終活動からの日数
}

# 時系列での活動変化を可視化
plot_activity_timeline(fp_developers)
```

---

### 3.2 False Negative（偽陰性）分析
**目的**: 承諾するのに承諾しないと予測してしまうケースを理解

**分析対象**:
- 予測: 承諾しない（label=0）
- 実際: 承諾する（label=1）

**仮説**:
1. **新規参入**: データが少なく過小評価
2. **復帰者**: 長期休止後の復帰
3. **突発的協力**: 普段は活動低いが特定イベントで参加
4. **メンター活動**: コードは書かないがレビューのみ活発

**分析手法**:
```python
# False Negative開発者の特徴
fn_analysis = {
    'developer_id': str,
    'prediction_score': float,  # 低スコアなのに実際は承諾
    'experience_days': int,  # 経験日数（新規?）
    'gap_before_acceptance': int,  # 前回活動からのギャップ
    'acceptance_trigger': str  # 承諾のきっかけ
}
```

---

## 4. モデル解釈性分析（Explainability）

### 4.1 特徴量重要度分析
**目的**: 各特徴量が予測にどれだけ寄与しているか

**分析手法**:
```python
# SHAP値計算（3モデル別）
import shap

for model in [nova, proj20, proj50]:
    explainer = shap.DeepExplainer(model, background_data)
    shap_values = explainer.shap_values(test_data)
    
    # 可視化
    shap.summary_plot(shap_values, features, feature_names)
    shap.dependence_plot('project_count', shap_values, features)
```

**期待される発見**:
- Nova: プロジェクト固有特徴が重要
- 20/50proj: プロジェクト横断特徴が重要
- project_count, cross_project_rate の寄与度

---

### 4.2 報酬関数の解釈
**目的**: IRLが学習した報酬関数を理解

**分析手法**:
```python
# 状態-報酬マッピング
state_reward_map = {
    'state_vector': np.array,
    'reward_score': float,
    'actual_continuation': bool
}

# 高報酬状態 vs 低報酬状態の比較
high_reward_states = filter(lambda x: x['reward'] > 0.8)
low_reward_states = filter(lambda x: x['reward'] < 0.2)

compare_state_distributions(high_reward_states, low_reward_states)
```

---

## 5. 実用的インサイト抽出

### 5.1 推奨レビュアー選定への応用
**目的**: モデル予測を実務に活かす方法を提案

**分析内容**:
```python
# シナリオ1: 新規レビュー依頼が来た場合
new_review = {
    'project': 'openstack/nova',
    'files_changed': 10,
    'lines_changed': 150,
    'author': 'developer_A'
}

# 候補レビュアーをスコアリング
candidate_reviewers = rank_reviewers(
    new_review,
    model=proj50_improved,
    top_k=10
)

# 推奨理由を説明
for reviewer in candidate_reviewers:
    explain_recommendation(reviewer, new_review)
```

**出力例**:
```
推奨レビュアー Top 5:
1. developer_123 (スコア: 0.92)
   - 理由: このファイルパスの専門家（過去10回レビュー）
   - 予測的中率: 95%
   - 平均応答時間: 2.3日
   
2. developer_456 (スコア: 0.88)
   - 理由: プロジェクト横断で活発（5プロジェクト活動）
   - 予測的中率: 90%
   - 平均応答時間: 1.8日
```

---

### 5.2 プロジェクト健全性指標
**目的**: プロジェクトのレビュー文化を定量評価

**指標定義**:
```python
project_health_metrics = {
    'project': str,
    'active_reviewers': int,  # 活発なレビュアー数
    'avg_acceptance_rate': float,  # 平均承諾率
    'cross_project_ratio': float,  # 横断開発者比率
    'predictability_score': float,  # 予測可能性（モデル精度）
    'bus_factor': int,  # バスファクター
    'knowledge_distribution': float  # 知識の分散度
}

# 健全性スコア算出
health_score = calculate_health_score(project_health_metrics)
```

**可視化**:
- プロジェクト別健全性レーダーチャート
- 時系列での健全性推移

---

## 6. 実装計画

### Phase 1: データ準備（1-2時間）
```bash
# Nova単体の結果を収集
# 20proj, 50proj, 50proj_improvedの結果を統合
# 予測結果CSVを統一フォーマットに変換
```

### Phase 2: 基本比較分析（2-3時間）
```python
scripts/analysis/compare_models.py
- 性能比較表作成
- 予測一致・不一致分析
- 基本的な可視化
```

### Phase 3: 開発者特性分析（3-4時間）
```python
scripts/analysis/analyze_developer_characteristics.py
- 的中率セグメンテーション
- 14次元状態特徴量分析
- 5次元行動特徴量分析
- プロジェクト横断活動分析
```

### Phase 4: 誤予測分析（2-3時間）
```python
scripts/analysis/analyze_prediction_errors.py
- False Positive深堀り
- False Negative深堀り
- エラーパターン分類
```

### Phase 5: 可視化とドキュメント（2-3時間）
```python
scripts/analysis/visualize_results.py
- 全グラフ生成
- インタラクティブダッシュボード（Plotly/Streamlit）
- 包括的分析レポート作成
```

### Phase 6: 実用的応用（2-3時間）
```python
scripts/analysis/practical_applications.py
- レビュアー推奨システムのプロトタイプ
- プロジェクト健全性ダッシュボード
```

---

## 7. 期待される成果物

### 7.1 分析レポート
- `docs/comprehensive_analysis_report.md`
  - 3モデル比較の全結果
  - 開発者特性の統計的分析
  - 実用的インサイト

### 7.2 可視化
- `outputs/visualizations/`
  - model_comparison_radar.png
  - prediction_agreement_venn.png
  - developer_segments_heatmap.png
  - feature_importance_shap.png
  - temporal_degradation_curves.png
  - error_analysis_sankey.png

### 7.3 データセット
- `outputs/analysis_data/`
  - model_comparison_summary.csv
  - developer_characteristics.csv
  - prediction_patterns.csv
  - error_cases.csv

### 7.4 インタラクティブツール
- `tools/reviewer_recommender/`
  - Streamlitダッシュボード
  - レビュアー推奨デモ

---

## 8. 分析で答えるべき研究課題（RQ）

### RQ1: プロジェクト数の増加は予測性能を向上させるか？
- Nova(1) vs 20proj vs 50proj
- F1, AUC-ROC, 時間的ロバスト性で評価

### RQ2: どのような開発者が予測しやすいか？
- 14次元状態特徴量との相関
- プロジェクト横断活動の影響

### RQ3: マルチプロジェクト学習の恩恵を受けるのはどのケースか？
- 専門型 vs 横断型開発者
- プロジェクト固有知識 vs 汎用パターン

### RQ4: モデルはどのように判断しているか？
- 報酬関数の解釈
- 特徴量重要度（SHAP）

### RQ5: 実務でどう活用できるか？
- レビュアー推奨の精度
- プロジェクト健全性モニタリング

---

**次のステップ**: 改善版学習の完了を待って、Phase 1から順次実装開始

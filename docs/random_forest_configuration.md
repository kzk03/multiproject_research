# Random Forest 設定詳細

## 使用特徴量（19次元）

Random ForestはIRLと**全く同じ特徴量**を使用しています。

### 14次元 状態特徴量（State Features）

| # | 特徴量名 | 説明 | データ型 |
|---|----------|------|----------|
| 1 | `experience_days` | 初回活動からの経験日数 | 数値 |
| 2 | `total_changes` | 総変更数（コミット数） | 数値 |
| 3 | `total_reviews` | 総レビュー数 | 数値 |
| 4 | `recent_activity_frequency` | 最近の活動頻度（回/日） | 数値 |
| 5 | `avg_activity_gap` | 平均活動間隔（日） | 数値 |
| 6 | `activity_trend` | 活動トレンド（増加/安定/減少） | 数値（1.0/0.0/-1.0） |
| 7 | `collaboration_score` | 協力スコア（レビュー・マージ比率） | 数値（0-1） |
| 8 | `code_quality_score` | コード品質スコア（テスト・ドキュメント比率） | 数値（0-1） |
| 9 | `recent_acceptance_rate` | 最近の受諾率（30日） | 数値（0-1） |
| 10 | `review_load` | レビュー負荷（レビュー/変更比） | 数値 |
| 11 | `project_count` | 参加プロジェクト数 | 数値 |
| 12 | `project_activity_distribution` | プロジェクト活動分散度 | 数値（0-1） |
| 13 | `main_project_contribution_ratio` | メインプロジェクト貢献率 | 数値（0-1） |
| 14 | `cross_project_collaboration_score` | クロスプロジェクト協力スコア | 数値（0-1） |

### 5次元 行動特徴量（Action Features）

| # | 特徴量名 | 説明 | データ型 |
|---|----------|------|----------|
| 15 | `avg_action_intensity` | 平均行動強度（変更サイズ） | 数値 |
| 16 | `avg_collaboration` | 平均協力度（レビュー参加率） | 数値（0-1） |
| 17 | `avg_response_time` | 平均応答時間（時間） | 数値 |
| 18 | `avg_review_size` | 平均レビューサイズ（行数） | 数値 |
| 19 | `cross_project_action_ratio` | クロスプロジェクト行動比率 | 数値（0-1） |

## Random Forest ハイパーパラメータ

比較実験では**3つの設定**を試しています：

### 1. Random Forest（標準）

```python
RandomForestClassifier(
    n_estimators=200,        # 決定木の数
    max_depth=20,            # 木の最大深さ
    min_samples_split=5,     # 分割に必要な最小サンプル数
    min_samples_leaf=2,      # 葉ノードの最小サンプル数
    class_weight='balanced', # クラス不均衡対策
    random_state=42,         # 再現性のためのシード
    n_jobs=-1                # 全CPUコア使用
)
```

**特徴**:
- バランスの取れた設定
- 過学習を防ぐため深さを制限（max_depth=20）

### 2. Random Forest (Deep)

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=None,          # 深さ制限なし（完全に分割）
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
```

**特徴**:
- より複雑なパターンを学習
- 過学習リスクが高い

### 3. Random Forest (More Trees)

```python
RandomForestClassifier(
    n_estimators=500,        # 木の数を2.5倍に増加
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
```

**特徴**:
- アンサンブル効果を強化
- 訓練時間が増加（約2.5倍）

## クラス不均衡対策

### `class_weight='balanced'`

クラス不均衡（正例90.2% vs 負例9.8%）に対応するため、自動的に重みを調整：

```python
weight_positive = n_samples / (n_classes * n_positive)
weight_negative = n_samples / (n_classes * n_negative)
```

**例**（183サンプル、正例165、負例18の場合）:
- 正例の重み: 183 / (2 * 165) = **0.55**
- 負例の重み: 183 / (2 * 18) = **5.08**

負例に約9倍の重みを与え、バランスを取る。

## 特徴量前処理

### activity_trendの数値変換

```python
trend_mapping = {
    'increasing': 1.0,   # 活動増加傾向
    'stable': 0.0,       # 活動安定
    'decreasing': -1.0   # 活動減少傾向
}
```

### 欠損値処理

```python
X = df[all_features].fillna(0)
```

すべての欠損値を0で埋める。

## IRLとの比較

| 項目 | IRL | Random Forest |
|------|-----|---------------|
| **特徴量** | 19次元（同じ） | 19次元（同じ） |
| **入力形式** | 時系列（seq_len=N） | スナップショット（単一時点） |
| **モデル** | LSTM + ニューラルネットワーク | アンサンブル学習（決定木） |
| **訓練時間** | 約4分（全10パターン） | 0.087秒（**100倍高速**） |
| **予測時間** | 約4秒（183サンプル） | 0.013秒（**300倍高速**） |
| **過学習** | 小サンプルで起きやすい | アンサンブルで抑制 |
| **解釈性** | 低い（ブラックボックス） | 高い（特徴量重要度が明確） |

## Random Forestの優位性

### 1. スナップショット特徴量に最適

**IRL時系列予測**:
- 3ヶ月間の時系列データ（seq_len=3～10）を使用
- 「月1→月2→月3」のトレンドを学習

**Random Forest**:
- 最終時点のスナップショットのみ使用
- 19個の特徴量から直接パターンを学習

**どちらも同じ19特徴量**だが、RFはスナップショット予測に特化。

### 2. 小サンプルでの安定性

**183サンプル**では：
- ニューラルネットワーク（IRL）: データ不足で過学習
- Random Forest: アンサンブル学習で汎化性能が高い

**一般的な推奨**:
- 1000サンプル以下: Random Forest
- 1000サンプル以上: ニューラルネットワーク（IRL）を検討

### 3. 計算効率

| 処理 | Random Forest | IRL |
|------|---------------|-----|
| 訓練 | 0.087秒 | 約240秒 |
| 予測（183サンプル） | 0.013秒 | 約4秒 |

**実用上の利点**:
- リアルタイム予測が可能
- グリッドサーチでのハイパーパラメータ調整が高速
- デプロイが簡単（依存関係が少ない）

### 4. 特徴量重要度の可視化

Random Forestは各特徴量の重要度を自動計算：

```python
feature_importance = rf_model.feature_importances_
```

**出力例**（上位5特徴）:
1. `avg_activity_gap`: 0.158（最重要）
2. `total_reviews`: 0.142
3. `avg_response_time`: 0.119
4. `experience_days`: 0.098
5. `project_count`: 0.087

IRLでは**Permutation Importance**を別途計算する必要がある。

## 実験結果（6-9m → 6-9m）

| モデル | F1 | AUC-ROC | Precision | Recall | Accuracy |
|--------|-----|---------|-----------|--------|----------|
| **Random Forest** | 0.997 | 0.999 | 1.000 | 0.994 | 0.995 |
| **RF (Deep)** | 0.997 | 0.999 | 1.000 | 0.994 | 0.995 |
| **RF (More Trees)** | 0.997 | 0.999 | 1.000 | 0.994 | 0.995 |
| IRL (時系列) | 0.944 | 0.728 | 0.923 | 0.966 | 0.938 |

**結論**: 3つのRF設定はほぼ同じ性能。**標準設定で十分**。

## 推奨設定

### 本プロジェクト（183サンプル）

```python
RandomForestClassifier(
    n_estimators=200,        # 標準で十分
    max_depth=20,            # 過学習防止
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced', # 必須（不均衡データ）
    random_state=42,
    n_jobs=-1
)
```

### データが増えた場合（1000サンプル以上）

```python
RandomForestClassifier(
    n_estimators=500,        # 木を増やす
    max_depth=30,            # より深い木
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
```

## まとめ

| 項目 | 内容 |
|------|------|
| **特徴量** | IRLと同じ19次元（14状態 + 5行動） |
| **入力** | スナップショット（最終時点のみ） |
| **設定** | 標準RF（n=200, depth=20） |
| **性能** | F1=0.997（IRLの0.944より+5.6%高い） |
| **速度** | 訓練0.087秒（IRLの100倍高速） |
| **推奨** | 現状の183サンプルでは**RF一択** |

---

**実装ファイル**: [scripts/analysis/compare_irl_vs_rf.py](scripts/analysis/compare_irl_vs_rf.py)
**実験日時**: 2025年12月15日

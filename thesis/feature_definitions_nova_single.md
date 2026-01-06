# Nova単体プロジェクトにおける特徴量定義

**注**: 本研究ではOpenStack/Nova単体プロジェクトを対象としているため、マルチプロジェクト特徴量（プロジェクト数、プロジェクト間活動分散度など）は使用していない。

---

## 1. 状態特徴量（State Features）: 10次元

レビュアーの現在の状態を表す特徴量。各時点での開発者の累積的な活動実績や傾向を捉える。

| # | 特徴量名 | 定義 | 正規化方法 | コード実装 |
|---|---------|------|-----------|----------|
| 1 | **経験日数**<br>(experience_days) | 初回活動から現在までの日数 | `min(days / 730.0, 1.0)`<br>2年(730日)でキャップ | `line 588` |
| 2 | **総コミット数**<br>(total_changes) | これまでの総コミット数 | `min(count / 500.0, 1.0)`<br>500件でキャップ | `line 589` |
| 3 | **総レビュー数**<br>(total_reviews) | これまでの総レビュー数 | `min(count / 500.0, 1.0)`<br>500件でキャップ | `line 590` |
| 4 | **最近の活動頻度**<br>(recent_activity_frequency) | 直近30日間の1日あたり平均活動量<br>= 直近30日の活動数 / 30 | 0-1の範囲で既に正規化済み | `line 591` |
| 5 | **平均活動間隔**<br>(avg_activity_gap) | タイムスタンプ間の平均日数<br>= Σ(次の活動 - 前の活動) / (活動数-1) | `min(days / 60.0, 1.0)`<br>60日でキャップ | `line 592` |
| 6 | **活動トレンド**<br>(activity_trend) | 直近30日と30-60日前の活動数比較<br>• 比率 > 1.2 → 'increasing' (1.0)<br>• 比率 < 0.8 → 'decreasing' (0.0)<br>• その他 → 'stable' (0.5) | カテゴリ変数をエンコード<br>increasing: 1.0<br>stable: 0.5<br>decreasing: 0.0<br>unknown: 0.25 | `line 593` |
| 7 | **協力スコア**<br>(collaboration_score) | 協力的な活動の割合<br>= 協力活動数 / 総活動数<br>協力活動: review, merge, collaboration, mentoring | 0-1の範囲で既に正規化済み | `line 594` |
| 8 | **コード品質スコア**<br>(code_quality_score) | 品質向上活動の割合<br>= 品質活動数 / 総活動数<br>品質活動: test, doc, refactor, fix | 0-1の範囲で既に正規化済み | `line 595` |
| 9 | **最近の受諾率**<br>(recent_acceptance_rate) | 直近30日のレビュー依頼受諾率<br>= 受諾数 / (受諾数 + 拒否数) | 0-1の範囲で既に正規化済み | `line 596` |
| 10 | **レビュー負荷**<br>(review_load) | 現在の負荷の相対値<br>= 直近30日の未完了レビュー数 / 通常の平均レビュー数 | 0-1の範囲で正規化済み<br>1.0以上はキャップ | `line 597` |

---

## 2. 行動特徴量（Action Features）: 4次元

レビュアーがレビュー依頼に対して取る具体的な行動を表す特徴量。

| # | 特徴量名 | 定義 | 正規化方法 | コード実装 |
|---|---------|------|-----------|----------|
| 1 | **強度（ファイル数）**<br>(intensity) | 1回のレビューで変更されたファイル数<br>ファイル数が多いほど高負荷 | 既に0-1の範囲で正規化済み<br>(ファイル数を適切にスケール) | `line 618` |
| 2 | **協力度**<br>(collaboration) | そのレビュイーをレビューした回数<br>協力関係の強さを示す | 既に0-1の範囲で正規化済み<br>(共同レビュー回数をスケール) | `line 619` |
| 3 | **応答速度**<br>(response_time → response_speed) | レビューリクエストから応答までの日数を素早さに変換<br>`速度 = 1.0 / (1.0 + 日数 / 3.0)`<br>• 即日応答 → ~1.0<br>• 3日 → ~0.5<br>• 遅い → ~0.0 | 0-1の範囲に変換済み<br>3日で約0.5になるよう設計 | `line 613, 620` |
| 4 | **レビュー規模（行数）**<br>(review_size) | 1回のレビューで変更された総行数<br>行数が多いほど高負荷 | 既に0-1の範囲で正規化済み<br>(変更行数を適切にスケール) | `line 621` |

---

## 3. 特徴量の正規化戦略

すべての特徴量は**0-1の範囲に正規化**されている。これにより：

1. **スケールの統一**: 異なる単位の特徴量を同じ尺度で扱える
2. **学習の安定化**: ニューラルネットワークの勾配が安定
3. **解釈性の向上**: 0-1の範囲で相対的な重要度を比較可能

### 正規化の2つのパターン

**パターン1: キャッピング（上限クリップ）**
- 例: `経験日数 = min(days / 730.0, 1.0)`
- 730日(2年)を超えても1.0に固定
- 外れ値の影響を抑制

**パターン2: 事前正規化**
- 例: `最近の活動頻度`（既に0-1）
- データ作成時に正規化済み
- そのまま使用

---

## 4. マルチプロジェクト特徴量（本研究では不使用）

以下の特徴量はマルチプロジェクト予測用に定義されているが、**Nova単体プロジェクトの本研究では使用していない**。

### 不使用の状態特徴量（4次元）
- `project_count`: 参加プロジェクト数
- `project_activity_distribution`: プロジェクト間の活動分散度
- `main_project_contribution_ratio`: メインプロジェクトへの貢献率
- `cross_project_collaboration_score`: プロジェクト横断協力スコア

### 不使用の行動特徴量（1次元）
- `is_cross_project`: プロジェクト横断的な行動フラグ

これらの特徴量はコード内でコメントアウトされており、Nova単体用モデル(`irl_predictor_nova_single.py`)では無視されている。

---

## 5. 特徴量の実装箇所

### Nova単体用実装
- **ファイル**: `/src/review_predictor/model/irl_predictor_nova_single.py`
- **状態変換**: `state_to_tensor()` メソッド (line 574-605)
- **行動変換**: `action_to_tensor()` メソッド (line 607-626)
- **次元設定**: `state_dim=10, action_dim=4` (line 82-83 in extract script)

### マルチプロジェクト用実装（参考）
- **ファイル**: `/src/review_predictor/model/irl_predictor.py`
- **次元設定**: `state_dim=14, action_dim=5`

---

## 6. データ構造

### DeveloperState (状態)
```python
@dataclass
class DeveloperState:
    developer_id: str
    experience_days: int
    total_changes: int
    total_reviews: int
    recent_activity_frequency: float
    avg_activity_gap: float
    activity_trend: str  # 'increasing', 'stable', 'decreasing', 'unknown'
    collaboration_score: float
    code_quality_score: float
    recent_acceptance_rate: float
    review_load: float
    timestamp: datetime
```

### DeveloperAction (行動)
```python
@dataclass
class DeveloperAction:
    action_type: str  # 'commit', 'review', 'merge', etc.
    intensity: float
    collaboration: float
    response_time: float  # 日数
    review_size: float
    timestamp: datetime
```

---

## 7. 卒論での表記

### 表4.2（状態特徴量）の正確な定義

| 特徴量名 | 定義 | 正規化 |
|---------|------|--------|
| 経験日数 | 初回活動から現在までの日数 | 2年(730日)でキャップ、0-1 |
| 総コミット数 | これまでの総コミット数 | 500件でキャップ、0-1 |
| 総レビュー数 | これまでの総レビュー数 | 500件でキャップ、0-1 |
| 最近の活動頻度 | 直近30日の1日あたり平均活動量 | 0-1（正規化済み） |
| 平均活動間隔 | タイムスタンプ間の平均日数 | 60日でキャップ、0-1 |
| 活動トレンド | 直近30日と30-60日前の活動比較（増加/安定/減少） | カテゴリ変数エンコード、0-1 |
| 協力スコア | 協力的な活動(review/merge等)の割合 | 0-1（正規化済み） |
| コード品質スコア | 品質向上活動(test/doc/refactor/fix)の割合 | 0-1（正規化済み） |
| 最近の受諾率 | 直近30日のレビュー依頼受諾率 | 0-1（正規化済み） |
| レビュー負荷 | 直近30日の未完了レビュー数 / 平均 | 0-1（正規化済み） |

### 表4.3（行動特徴量）の正確な定義

| 特徴量名 | 定義 | 正規化 |
|---------|------|--------|
| 強度 | レビューするファイル数 | 0-1（正規化済み） |
| 協力度 | そのレビュイーをレビューした回数 | 0-1（正規化済み） |
| 応答速度 | レビューリクエストから応答までの素早さ<br>1.0/(1.0+日数/3.0)で変換 | 0-1（日数を速度に変換） |
| レビュー規模 | 1回のレビューで変更された総行数 | 0-1（正規化済み） |

---

## 8. 重要な注意点

1. **Nova単体のみ**: 本研究は単一プロジェクト（OpenStack/Nova）を対象とするため、プロジェクト横断的な特徴量は使用していない。

2. **時系列考慮**: 状態特徴量は各時点での累積値、行動特徴量は各レビュー依頼ごとの値を表す。

3. **正規化の一貫性**: すべての特徴量が0-1に正規化されているため、異なるスケールの特徴を公平に比較可能。

4. **応答速度の変換**: 元の`response_time`（日数、大きい方が遅い）を`response_speed`（速度、大きい方が速い）に変換している点に注意。

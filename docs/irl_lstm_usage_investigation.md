# IRL LSTM使用状況調査レポート

## 調査背景

Random Forest比較実験でIRLが大敗した理由を調査するため、IRLがLSTMを使った時系列学習を実際に行っているかを検証。

## 調査結果

### ✅ 訓練時: LSTMを**使用している**

**コード箇所**: [src/review_predictor/model/irl_predictor.py:1342-1344](src/review_predictor/model/irl_predictor.py#L1342-L1344)

```python
# forward_all_stepsで報酬と継続確率を予測
predicted_reward, predicted_continuation = self.network.forward_all_steps(
    state_sequence, action_sequence, lengths, return_reward=True
)
```

**時系列データの構築**:
- 各月時点での状態を動的に計算（L1306-L1332）
- `state_sequence`: `[1, seq_len, state_dim]` の3Dテンソル（L1337）
- `action_sequence`: `[1, seq_len, action_dim]` の3Dテンソル（L1338）
- `forward_all_steps()`: LSTMで全ステップを処理

**訓練データ構造**:
```python
{
    'monthly_activity_histories': [[...], ...],  # 各月時点の活動履歴（LSTM用）
    'step_labels': [0, 1, 1, 0, ...],           # 各月の継続ラベル
}
```

### ❌ 予測時: LSTMを**使用していない**（スナップショットのみ）

**コード箇所**: [scripts/train/train_cross_temporal_multiproject.py:258-262](scripts/train/train_cross_temporal_multiproject.py#L258-L262)

```python
for traj in train_trajectories:
    developer = traj.get('developer', traj.get('developer_info', {}))
    result = irl_system.predict_continuation_probability_snapshot(
        developer,
        traj['activity_history'],
        traj['context_date']
    )
```

**評価時も同様**: [L318-L322](scripts/train/train_cross_temporal_multiproject.py#L318-L322)

```python
for traj in eval_trajectories:
    result = irl_system.predict_continuation_probability_snapshot(
        traj['developer'],
        traj['activity_history'],
        traj['context_date']
    )
```

**`predict_continuation_probability_snapshot`の実装**:
- `seq_len=1` の単一時点スナップショット
- LSTMは形式的に通るが、時系列情報を活用できていない

## 重大な発見: 訓練と予測のミスマッチ

### 訓練時の処理
1. 月次活動履歴を構築（`monthly_activity_histories`）
2. 各月の状態・行動をLSTMで時系列処理
3. 各月の継続確率を予測（`step_labels`）
4. Focal Lossで損失計算・バックプロパゲーション

### 予測時の処理
1. 最新の状態・行動のみ抽出
2. `seq_len=1` で単一時点として処理
3. **時系列情報を全く使用しない**

## この問題がIRLの低性能を説明する理由

### 1. LSTMの能力を活かせていない
- 訓練時: 「月1→月2→月3」の**時系列パターン**を学習
- 予測時: 「月3のみ」のスナップショットで予測
- **学習した時系列パターンを全く使えない**

### 2. Random Forestとの公平な比較になっていない
- **Random Forest**: スナップショット特徴19個で訓練・予測（一貫性あり）
- **IRL**: 時系列で訓練、スナップショットで予測（**ミスマッチ**）

### 3. 小サンプル（183件）での影響
- LSTMは時系列パターンを学習するため、より多くのデータが必要
- スナップショット予測では、高度な時系列学習が無意味になる
- Random Forestの方が小サンプルで安定している理由が明確

## 利用可能な時系列予測メソッド

### `predict_continuation_probability_trajectory` が存在

**コード箇所**: [src/review_predictor/model/irl_predictor.py](src/review_predictor/model/irl_predictor.py) (推定位置: L1450付近)

このメソッドは:
- 完全な時系列データを使用
- LSTMで全ステップを処理
- 訓練時と同じ方法で予測

**しかし、現在のコードでは使用されていない**

## 推奨される対応策

### オプション1: 時系列予測に切り替え（IRL本来の性能を検証）

**変更箇所**: `scripts/train/train_cross_temporal_multiproject.py`

```python
# 現在（スナップショット）
result = irl_system.predict_continuation_probability_snapshot(
    developer,
    traj['activity_history'],
    traj['context_date']
)

# 提案（時系列）
result = irl_system.predict_continuation_probability_trajectory(
    developer,
    traj['activity_history'],
    traj['context_date']
)
```

**期待される効果**:
- LSTMの時系列学習能力を活用
- 訓練と予測の一貫性が保たれる
- IRLの真の性能が測定できる

**懸念点**:
- 183サンプルでは時系列の恩恵が限定的かもしれない
- 計算コストが増加（おそらく数倍）

### オプション2: スナップショット学習に統一（現実的）

**変更箇所**: `src/review_predictor/model/irl_predictor.py`

訓練時も`seq_len=1`のスナップショットで学習するように変更。

**期待される効果**:
- 訓練と予測の一貫性が保たれる
- 小サンプルでの学習が安定
- Random Forestとの公平な比較が可能

**懸念点**:
- IRLの時系列学習能力を放棄することになる
- それならRandom Forestでよいのでは？

### オプション3: Random Forest採用（最も現実的）

**根拠**:
- F1: 0.997 vs 0.878 (+13.5%改善)
- Specialist精度: 98.1% vs 50.0% (+96%改善)
- 訓練時間: 0.087秒 vs 数分（100倍高速）
- 小サンプルでの安定性が高い

**推奨事項**:
1. IRLを時系列予測に修正して再実験
2. それでもRFに勝てない場合はRF採用
3. 将来的にデータが増えたら（1000件以上）IRLを再検討

## 次のステップ

### 即座に実行可能な検証

1. **`predict_continuation_probability_trajectory`に切り替え**
   ```bash
   # scripts/train/train_cross_temporal_multiproject.py を編集
   # L258, L318 の predict_continuation_probability_snapshot を
   # predict_continuation_probability_trajectory に変更

   # 2x OS (6-9m) モデルで再実験
   bash run_single_pattern.sh
   ```

2. **IRL vs RF 再比較**
   ```bash
   # 時系列予測版IRLで再実験
   uv run python scripts/analysis/compare_irl_vs_rf.py \
     --use-trajectory-prediction
   ```

3. **結果の確認**
   - F1スコアが改善されるか？
   - それでもRFに勝てるか？
   - 計算コストはどの程度増加するか？

## 結論

**IRLは訓練時にLSTMを使用しているが、予測時には使用していない。**

この訓練と予測のミスマッチが、IRLの低性能の主要因である可能性が高い。

時系列予測に切り替えることで、IRLの真の性能を測定できる。それでもRandom Forestに劣る場合は、小サンプルではRFが最適という結論になる。

---

**調査日時**: 2025年12月15日
**調査対象コード**:
- [src/review_predictor/model/irl_predictor.py](src/review_predictor/model/irl_predictor.py)
- [scripts/train/train_cross_temporal_multiproject.py](scripts/train/train_cross_temporal_multiproject.py)

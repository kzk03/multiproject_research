# RFベースライン 案1 vs 案2 - 詳細比較

## 目次
1. [概要](#概要)
2. [案1: IRL完全再現アプローチ](#案1-irl完全再現アプローチ)
3. [案2: シンプルベースラインアプローチ](#案2-シンプルベースラインアプローチ)
4. [詳細比較表](#詳細比較表)
5. [実装の具体例](#実装の具体例)
6. [推奨アプローチ](#推奨アプローチ)

---

## 概要

### 目的
IRLモデルと公平に比較できるRFベースラインを作成する

### 重要な比較軸
1. **訓練データの作り方**: スライディングウィンドウ vs 単一期間
2. **評価データの作り方**: IRLと完全一致 vs 簡略化
3. **実装の複雑さ**: 高 vs 低
4. **公平性**: IRLと同条件 vs 簡略化による不公平の可能性
5. **解釈のしやすさ**: 複雑 vs シンプル

---

## 案1: IRL完全再現アプローチ

### 基本方針
**IRLの訓練・評価プロセスを完全に再現し、モデルだけRFに置き換える**

### 訓練データ作成

#### ステップ1: 長期間データの準備
```python
# 訓練窓0-3mの場合
train_data_start = 2019-01-01  # IRLと同じ
train_data_end = 2021-04-01
label_window_months = 3  # 0-3m
```

#### ステップ2: スライディングウィンドウでサンプル生成
```python
samples = []

# 月次でスライディング
for cutoff_date in monthly_range(2019-01-01, 2021-01-01):
    # 履歴期間
    history_start = cutoff_date - 12ヶ月
    history_end = cutoff_date

    # ラベル期間
    label_start = cutoff_date
    label_end = cutoff_date + 3ヶ月  # 0-3m

    # この時点での開発者を抽出
    developers = get_active_developers(history_start, history_end)

    for dev in developers:
        # 特徴量計算
        features = extract_features(dev, history_start, history_end)

        # ラベル計算 (IRLと同じロジック)
        label = calculate_label_irl_logic(dev, label_start, label_end)

        if label is not None:  # 除外されない場合
            samples.append({
                'features': features,
                'label': label,
                'cutoff_date': cutoff_date
            })

# 訓練サンプル総数: 約1000～2000サンプル
print(f"訓練サンプル数: {len(samples)}")
```

**訓練サンプル数の例**:
- 時点数: 25ヶ月 (2019-01 ～ 2021-01)
- 各時点の平均開発者数: 50人
- 除外後の平均サンプル数: 40人/時点
- **総訓練サンプル数**: 25 × 40 = **1000サンプル**

#### ステップ3: Random Forest訓練
```python
from sklearn.ensemble import RandomForestClassifier

# IRLと同じ特徴量次元
# 状態10次元 + 行動4次元 = 14次元
X_train = [s['features'] for s in samples]  # shape: (1000, 14)
y_train = [s['label'] for s in samples]     # shape: (1000,)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
rf_model.fit(X_train, y_train)
```

### 評価データ作成

#### ステップ1: 評価期間の設定
```python
# 訓練窓0-3m → 評価窓0-3m の場合
train_cutoff_date = 2021-04-01
eval_period_start = 2023-01-01
eval_period_end = 2023-04-01
```

#### ステップ2: 履歴期間の計算 (IRLと同じ)
```python
# 方法A: 訓練終了日から12ヶ月前
history_start = train_cutoff_date - 12ヶ月  # 2020-04-01
history_end = train_cutoff_date  # 2021-04-01

# または
# 方法B: 評価開始日から12ヶ月前
history_start = eval_period_start - 12ヶ月  # 2022-01-01
history_end = eval_period_start  # 2023-01-01

# どちらを使うかはIRLの実装を確認して決定
```

#### ステップ3: 開発者抽出とラベル計算 (IRLと完全一致)
```python
# 履歴期間の開発者を抽出
developers = get_active_developers(history_start, history_end)
print(f"履歴期間の開発者数: {len(developers)}")  # 例: 80人

eval_samples = []

for dev in developers:
    # 最小依頼数フィルタ (IRLと同じ)
    if get_request_count(dev, history_start, history_end) < min_history_requests:
        continue

    # 評価期間のラベル計算 (IRLと完全同じロジック)
    label = calculate_label_irl_logic(
        dev,
        eval_period_start,
        eval_period_end,
        extended_period_end=eval_period_start + 12ヶ月
    )

    if label is None:  # 拡張期間にも依頼なし → 除外
        continue

    # 特徴量計算
    features = extract_features(dev, history_start, history_end)

    eval_samples.append({
        'features': features,
        'label': label,
        'developer': dev
    })

print(f"評価サンプル数: {len(eval_samples)}")  # 期待値: 60サンプル (0-3m)
```

#### ステップ4: 予測と評価
```python
X_eval = [s['features'] for s in eval_samples]
y_eval = [s['label'] for s in eval_samples]

# 予測確率
y_pred_proba = rf_model.predict_proba(X_eval)[:, 1]

# メトリクス計算
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

metrics = {
    'auc_roc': roc_auc_score(y_eval, y_pred_proba),
    'f1': f1_score(y_eval, (y_pred_proba > 0.5).astype(int)),
    'precision': precision_score(y_eval, (y_pred_proba > 0.5).astype(int)),
    'recall': recall_score(y_eval, (y_pred_proba > 0.5).astype(int))
}
```

### メリット

1. **✅ 完全な公平性**:
   - IRLと全く同じデータ・同じフィルタリング・同じ評価方法
   - 性能差はモデルの違いのみ

2. **✅ 正確な比較**:
   - サンプル数が完全一致 (eval 0-3m: 60サンプル)
   - 訓練データ量も同等

3. **✅ 論文での正当性**:
   - ベースラインとして最も厳密
   - レビュアーの批判を受けにくい

### デメリット

1. **❌ 実装が複雑**:
   - スライディングウィンドウの実装
   - IRLのロジックを完全理解する必要
   - デバッグが困難

2. **❌ 時間がかかる**:
   - 実装時間: 2-3日
   - 実行時間: 長い (1000サンプルの特徴量計算)

3. **❌ RFの利点を活かせない可能性**:
   - RFは単純な特徴量で強い
   - スライディングウィンドウで生成した時系列データは、RFには不向き

---

## 案2: シンプルベースラインアプローチ

### 基本方針
**訓練は簡略化し、評価だけIRLと一致させる**

### 訓練データ作成

#### ステップ1: 単一期間のデータ使用
```python
# 訓練窓0-3mの場合
train_period_start = 2021-01-01
train_period_end = 2021-04-01

# より長い期間を使うことも可能
# extended_train_start = 2019-01-01
# extended_train_end = 2021-04-01
```

#### ステップ2: 訓練サンプル作成 (シンプル版)
```python
# 方法A: 訓練期間のみ使用
history_start = train_period_start - 12ヶ月  # 2020-01-01
history_end = train_period_start  # 2021-01-01
label_start = train_period_start  # 2021-01-01
label_end = train_period_end  # 2021-04-01

# または
# 方法B: より長い訓練期間使用
history_start = 2019-01-01
history_end = 2021-01-01
label_start = 2021-01-01
label_end = 2021-04-01

# 開発者抽出
developers = get_active_developers(history_start, history_end)

train_samples = []
for dev in developers:
    # 最小依頼数フィルタ
    if get_request_count(dev, history_start, history_end) < min_history_requests:
        continue

    # ラベル計算 (IRLと同じロジック)
    label = calculate_label_irl_logic(dev, label_start, label_end)

    if label is None:
        continue

    features = extract_features(dev, history_start, history_end)

    train_samples.append({
        'features': features,
        'label': label
    })

print(f"訓練サンプル数: {len(train_samples)}")  # 例: 60-80サンプル
```

**訓練サンプル数の例**:
- 訓練期間の開発者数: 80人
- 除外後: **60サンプル**

**方法Bを使う場合**:
- 長い訓練期間 (2019-2021) から1回だけサンプル生成
- 訓練サンプル数: **約200サンプル** (より多いデータ)

#### ステップ3: Random Forest訓練
```python
X_train = [s['features'] for s in train_samples]
y_train = [s['label'] for s in train_samples]

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
rf_model.fit(X_train, y_train)
```

### 評価データ作成

**案1と完全に同じ**

評価データの作成とメトリクス計算は、案1と全く同じプロセスを使用。
これにより、評価の公平性を保つ。

### メリット

1. **✅ 実装がシンプル**:
   - スライディングウィンドウ不要
   - 実装時間: 半日～1日

2. **✅ デバッグしやすい**:
   - サンプル生成が単純
   - 問題の特定が容易

3. **✅ RFの強みを活かせる**:
   - 単純な特徴量とデータで高性能
   - 過学習のリスクが低い

4. **✅ 評価は公平**:
   - 評価データと評価方法はIRLと完全一致
   - サンプル数も一致 (60サンプル)

### デメリット

1. **❌ 訓練データ量が少ない**:
   - 60-200サンプル vs IRLの1000サンプル
   - 訓練データの公平性に欠ける

2. **❌ 時系列の学習なし**:
   - IRLは時間変化を学習
   - RFは単一時点のみ

3. **❌ 論文での正当性が弱い**:
   - 「訓練データが異なる」と批判される可能性
   - ただし、「RFの実用的なベースライン」として正当化可能

---

## 詳細比較表

| 項目 | 案1: IRL完全再現 | 案2: シンプルベースライン |
|------|-----------------|------------------------|
| **訓練データ作成** | | |
| データ期間 | 2019-2021 (長期) | 2021のみ or 2019-2021 |
| サンプル生成方法 | スライディングウィンドウ | 単一期間から1回 |
| 訓練サンプル数 | ~1000サンプル | 60-200サンプル |
| 時系列考慮 | ✅ あり | ❌ なし |
| | | |
| **評価データ作成** | | |
| 評価期間 | 2023-01 ~ 2023-04 (IRL一致) | 2023-01 ~ 2023-04 (IRL一致) |
| 履歴期間計算 | IRLと同じロジック | IRLと同じロジック |
| 開発者抽出 | IRLと同じロジック | IRLと同じロジック |
| ラベル計算 | IRLと同じロジック | IRLと同じロジック |
| 除外ロジック | IRLと同じ | IRLと同じ |
| 評価サンプル数 | 60 (IRL一致) | 60 (IRL一致) |
| | | |
| **実装** | | |
| 実装の複雑さ | ⭐⭐⭐⭐⭐ 高 | ⭐⭐ 低 |
| 実装時間 | 2-3日 | 半日-1日 |
| デバッグ難易度 | 難しい | 簡単 |
| コード行数 | ~500行 | ~200行 |
| | | |
| **公平性** | | |
| 訓練データ | ✅ 完全一致 | ⚠️ 簡略化 |
| 評価データ | ✅ 完全一致 | ✅ 完全一致 |
| 評価方法 | ✅ 完全一致 | ✅ 完全一致 |
| 総合公平性 | ⭐⭐⭐⭐⭐ 最高 | ⭐⭐⭐⭐ 高 |
| | | |
| **性能予測** | | |
| 訓練データ量 | 多い (有利) | 少ない (不利) |
| RFへの適合性 | やや不向き (時系列) | 向いている (単純) |
| 予想AUC-ROC | 0.70-0.80 | 0.75-0.85 |
| IRLとの差 | 小さい可能性 | 大きい可能性あり |
| | | |
| **論文での正当性** | | |
| ベースラインとしての妥当性 | ⭐⭐⭐⭐⭐ 完璧 | ⭐⭐⭐⭐ 良い |
| 批判への対応 | 「完全に同じ条件」 | 「実用的なRFベースライン」 |
| 推薦可能性 | 高い | 中-高 |

---

## 実装の具体例

### 案1の実装 (疑似コード)

```python
def train_rf_case1_full_replication():
    """案1: IRL完全再現"""

    # ========== 訓練データ作成 ==========
    train_samples = []

    # スライディングウィンドウで訓練サンプル生成
    for cutoff_month in range(2019-01, 2021-01):  # 25ヶ月
        cutoff_date = pd.Timestamp(cutoff_month)

        # 履歴期間
        history_start = cutoff_date - pd.DateOffset(months=12)
        history_end = cutoff_date

        # ラベル期間 (0-3m)
        label_start = cutoff_date
        label_end = cutoff_date + pd.DateOffset(months=3)

        # この時点での開発者を抽出
        history_df = df[(df['timestamp'] >= history_start) &
                       (df['timestamp'] < history_end)]
        developers = history_df['email'].unique()

        for dev in developers:
            # 最小依頼数フィルタ
            dev_history = history_df[history_df['email'] == dev]
            if len(dev_history) < MIN_HISTORY_REQUESTS:
                continue

            # 特徴量計算
            features = extract_14d_features(dev, history_start, history_end)

            # ラベル計算 (IRLロジック)
            label_df = df[(df['timestamp'] >= label_start) &
                         (df['timestamp'] < label_end) &
                         (df['email'] == dev)]

            if len(label_df) == 0:
                # 拡張期間チェック
                extended_end = label_start + pd.DateOffset(months=12)
                extended_df = df[(df['timestamp'] >= label_start) &
                                (df['timestamp'] < extended_end) &
                                (df['email'] == dev)]
                if len(extended_df) == 0:
                    continue  # 除外
                label = 0  # 負例
            else:
                # 承諾があるか
                label = 1 if (label_df['label'] == 1).any() else 0

            train_samples.append({
                'features': features,
                'label': label,
                'cutoff_date': cutoff_date,
                'developer': dev
            })

    print(f"訓練サンプル数: {len(train_samples)}")  # ~1000

    # ========== モデル訓練 ==========
    X_train = np.array([s['features'] for s in train_samples])
    y_train = np.array([s['label'] for s in train_samples])

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)

    # ========== 評価データ作成 (IRLと完全一致) ==========
    eval_samples = create_eval_samples_irl_aligned(
        df,
        train_cutoff_date=pd.Timestamp('2021-04-01'),
        eval_start=pd.Timestamp('2023-01-01'),
        eval_end=pd.Timestamp('2023-04-01')
    )

    print(f"評価サンプル数: {len(eval_samples)}")  # 60

    # ========== 評価 ==========
    X_eval = np.array([s['features'] for s in eval_samples])
    y_eval = np.array([s['label'] for s in eval_samples])

    y_pred_proba = rf.predict_proba(X_eval)[:, 1]

    metrics = {
        'auc_roc': roc_auc_score(y_eval, y_pred_proba),
        'f1': f1_score(y_eval, (y_pred_proba > 0.5).astype(int)),
        'precision': precision_score(y_eval, (y_pred_proba > 0.5).astype(int)),
        'recall': recall_score(y_eval, (y_pred_proba > 0.5).astype(int))
    }

    return metrics
```

### 案2の実装 (疑似コード)

```python
def train_rf_case2_simple_baseline():
    """案2: シンプルベースライン"""

    # ========== 訓練データ作成 (シンプル) ==========

    # 方法A: 短期訓練期間
    history_start = pd.Timestamp('2020-01-01')
    history_end = pd.Timestamp('2021-01-01')
    label_start = pd.Timestamp('2021-01-01')
    label_end = pd.Timestamp('2021-04-01')

    # または
    # 方法B: 長期訓練期間
    # history_start = pd.Timestamp('2019-01-01')
    # history_end = pd.Timestamp('2021-01-01')
    # label_start = pd.Timestamp('2021-01-01')
    # label_end = pd.Timestamp('2021-04-01')

    # 開発者抽出
    history_df = df[(df['timestamp'] >= history_start) &
                   (df['timestamp'] < history_end)]
    developers = history_df['email'].unique()

    train_samples = []
    for dev in developers:
        # 最小依頼数フィルタ
        dev_history = history_df[history_df['email'] == dev]
        if len(dev_history) < MIN_HISTORY_REQUESTS:
            continue

        # 特徴量計算
        features = extract_14d_features(dev, history_start, history_end)

        # ラベル計算 (IRLロジック)
        label_df = df[(df['timestamp'] >= label_start) &
                     (df['timestamp'] < label_end) &
                     (df['email'] == dev)]

        if len(label_df) == 0:
            # 拡張期間チェック
            extended_end = label_start + pd.DateOffset(months=12)
            extended_df = df[(df['timestamp'] >= label_start) &
                            (df['timestamp'] < extended_end) &
                            (df['email'] == dev)]
            if len(extended_df) == 0:
                continue  # 除外
            label = 0
        else:
            label = 1 if (label_df['label'] == 1).any() else 0

        train_samples.append({
            'features': features,
            'label': label,
            'developer': dev
        })

    print(f"訓練サンプル数: {len(train_samples)}")  # 60-200

    # ========== モデル訓練 ==========
    X_train = np.array([s['features'] for s in train_samples])
    y_train = np.array([s['label'] for s in train_samples])

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)

    # ========== 評価データ作成 (案1と同じ - IRLと完全一致) ==========
    eval_samples = create_eval_samples_irl_aligned(
        df,
        train_cutoff_date=pd.Timestamp('2021-04-01'),
        eval_start=pd.Timestamp('2023-01-01'),
        eval_end=pd.Timestamp('2023-04-01')
    )

    print(f"評価サンプル数: {len(eval_samples)}")  # 60

    # ========== 評価 (案1と同じ) ==========
    X_eval = np.array([s['features'] for s in eval_samples])
    y_eval = np.array([s['label'] for s in eval_samples])

    y_pred_proba = rf.predict_proba(X_eval)[:, 1]

    metrics = {
        'auc_roc': roc_auc_score(y_eval, y_pred_proba),
        'f1': f1_score(y_eval, (y_pred_proba > 0.5).astype(int)),
        'precision': precision_score(y_eval, (y_pred_proba > 0.5).astype(int)),
        'recall': recall_score(y_eval, (y_pred_proba > 0.5).astype(int))
    }

    return metrics
```

---

## 推奨アプローチ

### 段階的アプローチを推奨

#### フェーズ1: 案2で迅速に実装・検証
1. シンプルベースライン(案2)を実装
2. 評価サンプル数がIRLと一致することを確認 (60サンプル)
3. 性能を測定して、IRLとの差を確認
4. 実装時間: 半日～1日

**メリット**:
- 迅速にベースライン結果が得られる
- IRLとの性能差を早期に把握
- 問題があればすぐに修正可能

#### フェーズ2: 必要に応じて案1を実装
案2の結果を見て判断:

**ケースA: 案2で十分良い結果** (RF AUC 0.75+)
- 案2のままで論文に使用
- 「実用的なRFベースライン」として正当化
- 追加実装不要

**ケースB: 案2で性能が低すぎる** (RF AUC < 0.70)
- 訓練データ不足が原因の可能性
- 案1を実装してより公平な比較
- 実装時間: 2-3日追加

**ケースC: レビュアーから指摘**
- 論文レビュー時に「訓練データが異なる」と指摘される
- 案1を追加実験として実施
- リバイスで対応

### 最終推奨

**まずは案2から始める**

理由:
1. ✅ 実装が簡単で迅速
2. ✅ 評価は公平 (IRL一致)
3. ✅ 多くの場合、十分良い結果が出る
4. ✅ 必要なら案1に拡張可能

**案1は以下の場合のみ実装**:
- 案2の性能が極端に低い
- 論文レビューで指摘された
- より厳密な比較が研究目的

---

## 実装チェックリスト

### 案2実装チェックリスト

- [ ] データ読み込み: `review_requests_openstack_multi_5y_detail.csv`
- [ ] Nova単体フィルタ: `df[df['project'] == 'openstack/nova']`
- [ ] 訓練期間設定: 2021-01-01 ~ 2021-04-01 (または2019-2021)
- [ ] 評価期間設定: 2023-01-01 ~ 2023-04-01
- [ ] 履歴期間計算: 実装
- [ ] 最小依頼数フィルタ: `min_history_requests` (0, 1, or 3)
- [ ] ラベル計算ロジック: IRLと一致
- [ ] 拡張期間チェック: 12ヶ月
- [ ] 除外ロジック: IRLと一致
- [ ] 特徴量抽出: 14次元 (状態10 + 行動4)
- [ ] RF訓練: `RandomForestClassifier`
- [ ] 評価サンプル数確認: 60サンプル (0-3m)
- [ ] メトリクス計算: AUC-ROC, F1, Precision, Recall
- [ ] 10パターン評価: 全て実行
- [ ] 結果保存: JSON + CSV

### 案1追加実装チェックリスト

- [ ] 訓練期間拡張: 2019-01-01 ~ 2021-04-01
- [ ] スライディングウィンドウ実装: 月次
- [ ] 各時点でのサンプル生成
- [ ] 訓練サンプル数確認: ~1000サンプル
- [ ] その他は案2と同じ

---

**生成日時**: 2025-12-19
**作成者**: Claude Code Analysis
**目的**: RFベースライン2案の詳細比較と推奨アプローチの提示

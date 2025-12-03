# プロジェクト技術ガイド

## レビュー承諾予測システム - 逆強化学習(IRL)ベース

> **目的**: Gerrit（コードレビューシステム）のレビュー依頼データから、レビュアーが将来もレビュー依頼を承諾し続けるかどうかを予測するシステム

---

## 📁 ファイル構成（7 ファイル）

```
gerrit-retention/
├── scripts/
│   ├── pipeline/
│   │   └── build_dataset.py      # ① データ収集 + 特徴量生成
│   ├── train/
│   │   └── train_model.py        # ② 訓練 + 評価
│   └── evaluate/
│       ├── cross_evaluate.py     # ③ クロス評価の実行
│       └── create_heatmaps.py    # ④ 評価結果の可視化
└── src/review_predictor/
    ├── __init__.py
    └── model/
        ├── __init__.py
        └── irl_predictor.py      # ⑤ IRLモデル本体
```

---

## 🔄 全体フロー図

```
┌─────────────────────────────────────────────────────────────────┐
│                      1. データ収集                              │
│   build_dataset.py                                               │
│   ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    │
│   │ Gerrit API  │───▶│ 変更データ   │───▶│ レビュー依頼    │    │
│   │ からFetch   │    │ (Changes)    │    │ 抽出            │    │
│   └─────────────┘    └──────────────┘    └────────┬────────┘    │
│                                                    │             │
│   ┌─────────────────────────────────────┐         │             │
│   │        特徴量計算                    │◀────────┘             │
│   │  - 履歴ベース（過去のレビュー数等）  │                        │
│   │  - パス類似度（ファイルの専門性）    │                        │
│   │  - インタラクション履歴             │                        │
│   └────────────────────┬────────────────┘                        │
│                        ▼                                         │
│              data/xxx_dataset.csv                                │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      2. 訓練・評価                               │
│   train_model.py + irl_predictor.py                             │
│                                                                  │
│   ┌─────────────────┐    ┌─────────────────┐                    │
│   │ 軌跡抽出        │───▶│ IRL訓練         │                    │
│   │ (Trajectories)  │    │ (LSTM + Focal)  │                    │
│   └─────────────────┘    └────────┬────────┘                    │
│                                   │                              │
│   ┌─────────────────┐             │                              │
│   │ 評価用軌跡抽出  │◀────────────┘                              │
│   │ (Snapshot)      │                                            │
│   └────────┬────────┘                                            │
│            │                                                     │
│   ┌────────▼────────┐    ┌─────────────────┐                    │
│   │ 予測実行        │───▶│ メトリクス計算  │                    │
│   │                 │    │ (AUC, F1等)     │                    │
│   └─────────────────┘    └─────────────────┘                    │
│                                                                  │
│   出力: outputs/xxx/                                             │
│   ├── irl_model.pt              # 学習済みモデル                 │
│   ├── optimal_threshold.json    # 最適閾値                       │
│   ├── metrics.json              # 評価メトリクス                 │
│   └── predictions.csv           # 予測結果                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📄 各ファイルの詳細説明

### 1️⃣ `scripts/pipeline/build_dataset.py` (669 行)

**役割**: Gerrit API からデータを取得し、特徴量付き CSV を生成

**呼び出し方**:

```bash
uv run python scripts/pipeline/build_dataset.py \
    --gerrit-url https://review.opendev.org \
    --project openstack/nova \
    --start-date 2020-01-01 \
    --end-date 2024-01-01 \
    --output data/nova_dataset.csv
```

**クラス構成**:

#### `GerritDataFetcher` (47-121 行)

```
目的: Gerrit REST APIからデータ取得
メソッド:
├── __init__(gerrit_url, timeout)
│   └── セッション初期化、URL設定
├── _make_request(endpoint, params)
│   └── APIリクエスト実行、Gerrit形式レスポンス解析
└── fetch_changes(project, start_date, end_date, limit)
    └── 変更データをページネーションで全件取得
```

#### `FeatureBuilder` (124-530 行)

```
目的: 変更データから特徴量を計算
メソッド:
├── __init__(response_window_days, bot_patterns)
│   └── 応答ウィンドウ(14日)、ボットパターン設定
├── _is_bot(email)
│   └── ボット判定（zuul, jenkins等を除外）
├── _extract_review_requests(changes)
│   └── 変更→レビュー依頼を抽出
│       ・reviewersフィールドから明示的レビュアー
│       ・messagesから実際に応答したレビュアー
│       ・応答の有無でラベル付け（14日以内=1, それ以外=0）
├── _compute_history_features(requests, changes)
│   └── 履歴ベース特徴量を時系列順に計算（データリーク防止）
└── _compute_path_similarity(reviewer, project, files, ...)
    └── パス類似度特徴量（Jaccard, Dice, Overlap, Cosine）
```

**生成される特徴量（約 65 種類）**:

```
基本情報:
├── change_id, project, owner_email, reviewer_email
├── request_time, label (1=承諾, 0=拒否)
└── response_latency_days

履歴ベース:
├── reviewer_past_reviews_30d/90d/180d    # 過去のレビュー数
├── owner_past_messages_30d/90d/180d      # オーナーの活動
├── owner_reviewer_past_interactions_180d # 過去のやりとり
├── reviewer_assignment_load_7d/30d/180d  # レビュー負荷
├── reviewer_past_response_rate_180d      # 過去の応答率
└── reviewer_tenure_days                   # 在籍日数

パス類似度:
├── path_jaccard_files_global/project     # Jaccard係数
├── path_dice_files_global/project        # Dice係数
├── path_overlap_coeff_files_global       # Overlap係数
└── path_cosine_files_global              # Cosine類似度
```

---

### 2️⃣ `scripts/train/train_model.py` (965 行)

**役割**: IRL モデルの訓練と評価を実行

**呼び出し方**:

```bash
uv run python scripts/train/train_model.py \
    --reviews data/nova_dataset.csv \
    --train-start 2021-01-01 \
    --train-end 2023-01-01 \
    --eval-start 2023-01-01 \
    --eval-end 2024-01-01 \
    --future-window-start 0 \
    --future-window-end 3 \
    --epochs 20 \
    --output outputs/nova_model
```

**引数の意味**:
| 引数 | 説明 | デフォルト |
|------|------|-----------|
| `--reviews` | 入力 CSV パス | 必須 |
| `--train-start/end` | 訓練期間 | 必須 |
| `--eval-start/end` | 評価期間 | 必須 |
| `--future-window-start/end` | 将来窓（月） | 0-3 |
| `--epochs` | 訓練エポック数 | 20 |
| `--min-history-events` | 最小履歴イベント数 | 3 |
| `--project` | プロジェクト絞り込み | None |
| `--model` | 既存モデルパス（評価のみ） | None |

**主要関数**:

#### `extract_review_acceptance_trajectories()` (88-330 行)

```
目的: 訓練用の軌跡（Trajectory）を抽出
入力: DataFrameと期間
出力: 軌跡リスト（各レビュアー1サンプル）

処理フロー:
1. 特徴量計算期間のデータを抽出
2. 各レビュアーについて:
   ├── 最小レビュー依頼数を確認
   ├── ラベル計算期間のデータを抽出
   ├── 継続判定:
   │   ├── 承諾あり → 正例 (label=1)
   │   ├── 依頼あり・全拒否 → 負例 (label=0, weight=1.0)
   │   └── 依頼なし・拡張期間に依頼あり → 負例 (weight=0.1)
   └── 月次ラベル(step_labels)を計算
       └── 各月時点から将来窓を見てラベル付け

出力構造:
{
    'developer_info': {...},
    'activity_history': [...],           # 全期間の活動履歴
    'monthly_activity_histories': [...], # 各月時点の活動履歴（LSTM用）
    'step_labels': [0, 1, 1, 0, ...],    # 月次継続ラベル
    'future_acceptance': True/False,     # 最終ラベル
    'sample_weight': 1.0/0.1             # サンプル重み
}
```

#### `extract_evaluation_trajectories()` (333-565 行)

```
目的: 評価用の軌跡を抽出（スナップショット特徴量用）
特徴: 訓練とは別のロジックで、cutoff時点の状態のみ使用

処理フロー:
1. 履歴期間 = cutoff - history_window ～ cutoff
2. 評価期間 = cutoff + future_window_start ～ cutoff + future_window_end
3. 各レビュアーについて:
   ├── 履歴期間のデータを抽出
   └── 評価期間での継続を判定
```

#### `main()` (620-965 行)

```
訓練・評価のメインフロー:

1. データ読み込み
2. 訓練用軌跡を抽出
3. IRLシステムを初期化
   └── config = {state_dim: 10, action_dim: 4, hidden_dim: 128, ...}
4. Focal Lossを正例率に応じて自動調整
5. 訓練実行
6. 訓練データで最適閾値を決定（F1最大化）
7. モデル・閾値を保存
8. 評価用軌跡を抽出
9. 予測実行
10. メトリクス計算・保存
```

---

### 3️⃣ `src/review_predictor/model/irl_predictor.py` (1242 行)

**役割**: 逆強化学習(IRL)モデルの実装

**クラス構成**:

#### `DeveloperState` (22-36 行)

```python
@dataclass
class DeveloperState:
    """開発者の状態表現（10次元）"""
    developer_id: str
    experience_days: int           # 経験日数
    total_changes: int             # コミット総数
    total_reviews: int             # レビュー総数
    recent_activity_frequency: float  # 直近30日の活動頻度
    avg_activity_gap: float        # 平均活動間隔（日）
    activity_trend: str            # 活動トレンド（increasing/stable/decreasing）
    collaboration_score: float     # 協力スコア
    code_quality_score: float      # コード品質スコア
    recent_acceptance_rate: float  # 直近30日のレビュー受諾率 ✨
    review_load: float             # レビュー負荷 ✨
    timestamp: datetime
```

#### `DeveloperAction` (39-49 行)

```python
@dataclass
class DeveloperAction:
    """開発者の行動表現（5次元）"""
    action_type: str      # 行動タイプ（review等）
    intensity: float      # 強度（変更ファイル数ベース）
    collaboration: float  # 協力度
    response_time: float  # レスポンス時間（日）
    review_size: float    # レビュー規模（変更行数）✨
    timestamp: datetime
```

#### `RetentionIRLNetwork` (52-165 行)

```
ニューラルネットワーク構造:

入力:
├── state: [batch, seq_len, 10]   # 状態（10次元）
└── action: [batch, seq_len, 5]   # 行動（5次元）

アーキテクチャ:
├── state_encoder: Linear(10→128) → ReLU → Dropout → Linear(128→64) → ReLU
├── action_encoder: Linear(5→128) → ReLU → Dropout → Linear(128→64) → ReLU
├── LSTM: input=64, hidden=128, layers=1
├── reward_predictor: Linear(128→64) → ReLU → Linear(64→1)
└── continuation_predictor: Linear(128→64) → ReLU → Linear(64→1) → Sigmoid

出力:
├── reward: [batch, 1]            # 報酬スコア
└── continuation_prob: [batch, 1] # 継続確率（0-1）
```

**forward 処理の 2 モード**:

1. **シーケンスモード** (`forward()` 97-145 行)

```
時系列データを処理:
state_encoded + action_encoded → LSTM → hidden → reward/continuation
可変長対応: pack_padded_sequence使用
```

2. **全ステップ予測** (`forward_all_steps()` 167-218 行)

```
全タイムステップで継続確率を出力:
→ LSTM出力を各ステップで予測器に通す
→ [batch, seq_len] の予測確率を返す
```

#### `RetentionIRLSystem` (220-1242 行)

```
メインのIRLシステムクラス

初期化パラメータ:
├── state_dim: 10
├── action_dim: 5
├── hidden_dim: 128
├── sequence: True（LSTMモード）
├── seq_len: 0（可変長）
├── dropout: 0.1-0.2
└── learning_rate: 0.0001-0.0003

主要メソッド:

特徴量抽出:
├── extract_developer_state()     # 状態抽出（10次元）
├── extract_developer_actions()   # 行動抽出（5次元）
├── state_to_tensor()             # 状態→テンソル変換（正規化）
└── action_to_tensor()            # 行動→テンソル変換（正規化）

訓練:
├── train_irl_temporal_trajectories()  # 時系列IRL訓練
│   └── 各軌跡について:
│       1. 月次活動履歴から状態・行動を計算
│       2. forward_all_stepsで予測
│       3. Focal Lossで損失計算
│       4. バックプロパゲーション
├── focal_loss()                  # Focal Loss計算
├── auto_tune_focal_loss()        # 正例率に応じたパラメータ自動調整
└── set_focal_loss_params()       # Focal Lossパラメータ設定

予測:
├── predict_continuation_probability()          # 時系列予測
└── predict_continuation_probability_snapshot() # スナップショット予測（評価用）
```

**Focal Loss** (273-300 行):

```
目的: クラス不均衡対策
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

パラメータ自動調整（正例率に応じて）:
├── ≥60%: α=0.40, γ=1.0（バランス重視）
├── 30-60%: α=0.45, γ=1.0（継続重視）
└── <30%: α=0.55, γ=1.1（継続重視・強）
```

---

### 4️⃣ `scripts/evaluate/cross_evaluate.py` (129 行)

**役割**: 訓練期間 × 評価期間のクロス評価を実行

**処理フロー**:

```
1. 訓練期間ごとにモデルを訓練
   train_periods = ['0-3m', '3-6m', '6-9m', '9-12m']

   各期間でtrain_model.pyを実行:
   └── outputs/cross_eval/train_{period}/irl_model.pt

2. 各訓練モデルで全評価期間を評価
   eval_periods = ['0-3m', '3-6m', '6-9m', '9-12m']

   16通りの組み合わせで評価:
   └── outputs/cross_eval/train_{train_period}/eval_{eval_period}/metrics.json
```

---

### 5️⃣ `scripts/evaluate/create_heatmaps.py` (129 行)

**役割**: クロス評価結果をヒートマップで可視化

**処理**:

```
1. 全組み合わせのmetrics.jsonを読み込み
2. 5種類のメトリクスでヒートマップ作成
   ├── AUC-ROC
   ├── AUC-PR
   ├── F1 Score
   ├── Precision
   └── Recall
3. 最高値にマーカー表示
4. PNGファイルとして保存
```

---

## 🏷️ ラベルの定義

### レビュー依頼ラベル（build_dataset.py）

```
label = 1: レビュー依頼に対して14日以内に応答した（承諾）
label = 0: レビュー依頼に対して14日以内に応答しなかった（拒否）
```

### 継続ラベル（train_model.py）

```
訓練時（step_labels）:
- 各月末時点から将来窓を見て判定
- 将来窓内に承諾があれば1、なければ0

評価時（future_acceptance）:
- 評価期間内に少なくとも1つ承諾 → 正例（継続）
- 評価期間内に依頼あり・全拒否 → 負例（離脱）
- 依頼なし → 除外（予測対象外）
```

---

## 🚀 クイックスタート（5 分で始める）

### 前提条件

```bash
# Python 3.10以上が必要
python --version

# uvがインストールされていることを確認
uv --version

# プロジェクトディレクトリに移動
cd /path/to/gerrit-retention

# 依存関係をインストール（初回のみ）
uv sync
```

### 最小限の実行例

```bash
# 1. 既存データで訓練・評価（約1分）
uv run python scripts/train/train_model.py \
    --reviews data/review_requests_openstack_multi_5y_detail.csv \
    --train-start 2021-01-01 \
    --train-end 2023-01-01 \
    --eval-start 2023-01-01 \
    --eval-end 2024-01-01 \
    --epochs 10 \
    --output outputs/quick_test

# 2. 結果を確認
cat outputs/quick_test/metrics.json
```

---

## 📋 詳細な実行手順

### ステップ 1: 環境セットアップ

```bash
# 1-1. リポジトリをクローン
git clone https://github.com/kzk03/rl.git
cd rl/gerrit-retention

# 1-2. Python仮想環境を作成（uvを使用）
uv venv
source .venv/bin/activate  # Linuxの場合
# または
source .venv/bin/activate.fish  # fishシェルの場合

# 1-3. 依存関係をインストール
uv sync

# 1-4. インストール確認
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}')"
uv run python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
```

### ステップ 2: データ収集（新規プロジェクトの場合）

```bash
# 2-1. Gerrit APIからデータを取得
# 注意: 大規模プロジェクトは数分〜数十分かかる場合があります

# 例: OpenStack Novaプロジェクト（4年分）
uv run python scripts/pipeline/build_dataset.py \
    --gerrit-url https://review.opendev.org \
    --project openstack/nova \
    --start-date 2020-01-01 \
    --end-date 2024-01-01 \
    --output data/nova_4years.csv

# 例: 複数プロジェクト同時取得
uv run python scripts/pipeline/build_dataset.py \
    --gerrit-url https://review.opendev.org \
    --project openstack/nova openstack/neutron openstack/cinder \
    --start-date 2020-01-01 \
    --end-date 2024-01-01 \
    --output data/openstack_multi_4years.csv

# 例: 短期間でテスト（2ヶ月）
uv run python scripts/pipeline/build_dataset.py \
    --gerrit-url https://review.opendev.org \
    --project openstack/neutron \
    --start-date 2024-09-01 \
    --end-date 2024-11-01 \
    --output data/neutron_test_2months.csv
```

**出力の確認**:

```bash
# 2-2. 生成されたデータを確認
head -5 data/nova_4years.csv
wc -l data/nova_4years.csv  # 行数確認

# 2-3. ラベル分布を確認
uv run python -c "
import pandas as pd
df = pd.read_csv('data/nova_4years.csv')
print(f'総レビュー依頼数: {len(df)}')
print(f'承諾率: {df[\"label\"].mean()*100:.1f}%')
print(f'レビュアー数: {df[\"reviewer_email\"].nunique()}')
"
```

### ステップ 3: モデル訓練

```bash
# 3-1. 基本的な訓練
uv run python scripts/train/train_model.py \
    --reviews data/nova_4years.csv \
    --train-start 2021-01-01 \
    --train-end 2023-01-01 \
    --eval-start 2023-01-01 \
    --eval-end 2024-01-01 \
    --epochs 20 \
    --output outputs/nova_model

# 3-2. パラメータ調整版（推奨設定）
uv run python scripts/train/train_model.py \
    --reviews data/nova_4years.csv \
    --train-start 2021-01-01 \
    --train-end 2023-01-01 \
    --eval-start 2023-01-01 \
    --eval-end 2024-01-01 \
    --future-window-start 0 \
    --future-window-end 3 \
    --epochs 30 \
    --min-history-events 5 \
    --output outputs/nova_model_optimized

# 3-3. 単一プロジェクト絞り込み
uv run python scripts/train/train_model.py \
    --reviews data/openstack_multi_4years.csv \
    --train-start 2021-01-01 \
    --train-end 2023-01-01 \
    --eval-start 2023-01-01 \
    --eval-end 2024-01-01 \
    --project openstack/nova \
    --epochs 30 \
    --output outputs/nova_only_model
```

**訓練ログの見方**:

```
2025-12-03 14:40:40,685 - INFO - 訓練データ正例率: 44.4% (60/135)
2025-12-03 14:40:40,685 - INFO - Focal Loss パラメータ更新: alpha=0.450, gamma=1.000
2025-12-03 14:40:41,069 - INFO - エポック 0: 平均損失 = 1.1581
...
2025-12-03 14:40:44,339 - INFO - 時系列IRL訓練完了
2025-12-03 14:40:44,412 - INFO - F1最大化閾値（訓練データ）: 0.4909
2025-12-03 14:40:44,412 - INFO - 訓練データ性能: Precision=0.457, Recall=0.983, F1=0.624
```

### ステップ 4: 結果の確認

```bash
# 4-1. メトリクスを確認
cat outputs/nova_model/metrics.json

# 4-2. 詳細な結果をPythonで確認
uv run python -c "
import json
import pandas as pd

# メトリクス読み込み
with open('outputs/nova_model/metrics.json') as f:
    metrics = json.load(f)

print('=' * 50)
print('評価結果サマリ')
print('=' * 50)
print(f'AUC-ROC: {metrics[\"auc_roc\"]:.4f}')
print(f'AUC-PR: {metrics[\"auc_pr\"]:.4f}')
print(f'F1 Score: {metrics[\"f1_score\"]:.4f}')
print(f'Precision: {metrics[\"precision\"]:.4f}')
print(f'Recall: {metrics[\"recall\"]:.4f}')
print(f'正例数: {metrics[\"positive_count\"]}')
print(f'負例数: {metrics[\"negative_count\"]}')

# 予測結果を確認
preds = pd.read_csv('outputs/nova_model/predictions.csv')
print()
print('予測確率の分布:')
print(preds['predicted_prob'].describe())
"

# 4-3. 個別の予測結果を確認
head -20 outputs/nova_model/predictions.csv
```

### ステップ 5: 既存モデルで新データを評価

```bash
# 5-1. 既存モデルを使って別期間を評価
uv run python scripts/train/train_model.py \
    --reviews data/nova_4years.csv \
    --train-start 2021-01-01 \
    --train-end 2023-01-01 \
    --eval-start 2023-06-01 \
    --eval-end 2024-01-01 \
    --model outputs/nova_model/irl_model.pt \
    --output outputs/nova_model_eval_2023h2

# 5-2. 異なる将来窓で評価
uv run python scripts/train/train_model.py \
    --reviews data/nova_4years.csv \
    --train-start 2021-01-01 \
    --train-end 2023-01-01 \
    --eval-start 2023-01-01 \
    --eval-end 2024-01-01 \
    --future-window-start 3 \
    --future-window-end 6 \
    --model outputs/nova_model/irl_model.pt \
    --output outputs/nova_model_eval_3to6m
```

### ステップ 6: クロス評価（オプション）

```bash
# 6-1. クロス評価を実行
uv run python scripts/evaluate/cross_evaluate.py

# 6-2. ヒートマップを生成
uv run python scripts/evaluate/create_heatmaps.py

# 6-3. 結果を確認
ls outputs/cross_eval/
open outputs/cross_eval/all_metrics_heatmaps.png  # macOSの場合
```

---

## 🔧 よくある使用パターン

### パターン 1: 新規プロジェクトの完全なパイプライン

```bash
# データ収集 → 訓練 → 評価 の一連の流れ
PROJECT="openstack/neutron"
OUTPUT_NAME="neutron"

# データ収集
uv run python scripts/pipeline/build_dataset.py \
    --gerrit-url https://review.opendev.org \
    --project $PROJECT \
    --start-date 2020-01-01 \
    --end-date 2024-01-01 \
    --output data/${OUTPUT_NAME}_dataset.csv

# 訓練・評価
uv run python scripts/train/train_model.py \
    --reviews data/${OUTPUT_NAME}_dataset.csv \
    --train-start 2021-01-01 \
    --train-end 2023-01-01 \
    --eval-start 2023-01-01 \
    --eval-end 2024-01-01 \
    --epochs 30 \
    --output outputs/${OUTPUT_NAME}_model

# 結果確認
cat outputs/${OUTPUT_NAME}_model/metrics.json
```

### パターン 2: 複数の将来窓で比較

```bash
# 0-3ヶ月, 3-6ヶ月, 6-9ヶ月で訓練・評価を比較
for START in 0 3 6; do
    END=$((START + 3))
    uv run python scripts/train/train_model.py \
        --reviews data/nova_4years.csv \
        --train-start 2021-01-01 \
        --train-end 2023-01-01 \
        --eval-start 2023-01-01 \
        --eval-end 2024-01-01 \
        --future-window-start $START \
        --future-window-end $END \
        --epochs 30 \
        --output outputs/nova_window_${START}to${END}m
done

# 結果比較
for START in 0 3 6; do
    END=$((START + 3))
    echo "=== ${START}-${END}m ==="
    cat outputs/nova_window_${START}to${END}m/metrics.json | grep -E "auc_roc|f1_score"
done
```

### パターン 3: バッチ処理（複数プロジェクト）

```bash
# 複数プロジェクトを順次処理
PROJECTS=("openstack/nova" "openstack/neutron" "openstack/cinder")

for PROJECT in "${PROJECTS[@]}"; do
    NAME=$(echo $PROJECT | tr '/' '_')
    echo "Processing $PROJECT..."

    uv run python scripts/pipeline/build_dataset.py \
        --gerrit-url https://review.opendev.org \
        --project $PROJECT \
        --start-date 2022-01-01 \
        --end-date 2024-01-01 \
        --output data/${NAME}_2years.csv

    uv run python scripts/train/train_model.py \
        --reviews data/${NAME}_2years.csv \
        --train-start 2022-01-01 \
        --train-end 2023-06-01 \
        --eval-start 2023-06-01 \
        --eval-end 2024-01-01 \
        --epochs 20 \
        --output outputs/${NAME}_model
done
```

---

## ❓ トラブルシューティング

### 問題 1: データ取得が遅い

```bash
# 解決策: 期間を短くしてテスト
uv run python scripts/pipeline/build_dataset.py \
    --gerrit-url https://review.opendev.org \
    --project openstack/nova \
    --start-date 2024-01-01 \
    --end-date 2024-03-01 \
    --output data/nova_test.csv
```

### 問題 2: メモリ不足

```bash
# 解決策: バッチサイズを調整（train_model.pyの内部設定）
# または、短い期間で訓練
uv run python scripts/train/train_model.py \
    --reviews data/dataset.csv \
    --train-start 2022-01-01 \
    --train-end 2023-01-01 \
    --epochs 10 \
    --output outputs/small_model
```

### 問題 3: AUC-ROC が nan になる

```
原因: 評価データに正例または負例のみ
解決策: 評価期間を長くするか、データを増やす
```

```bash
# より長い評価期間を設定
uv run python scripts/train/train_model.py \
    --reviews data/dataset.csv \
    --train-start 2021-01-01 \
    --train-end 2023-01-01 \
    --eval-start 2023-01-01 \
    --eval-end 2024-06-01 \
    --output outputs/longer_eval
```

### 問題 4: 訓練用軌跡が抽出できない

```
原因: 最小履歴イベント数を満たすレビュアーがいない
解決策: min-history-eventsを下げる
```

```bash
uv run python scripts/train/train_model.py \
    --reviews data/dataset.csv \
    --train-start 2021-01-01 \
    --train-end 2023-01-01 \
    --min-history-events 2 \
    --output outputs/model
```

---

## 🔧 使用例（旧セクション - 互換性のため残す）

### 新規プロジェクトでの完全なパイプライン

```bash
# 1. データ収集
uv run python scripts/pipeline/build_dataset.py \
    --gerrit-url https://review.opendev.org \
    --project openstack/neutron \
    --start-date 2020-01-01 \
    --end-date 2024-01-01 \
    --output data/neutron_dataset.csv

# 2. 訓練・評価
uv run python scripts/train/train_model.py \
    --reviews data/neutron_dataset.csv \
    --train-start 2021-01-01 \
    --train-end 2023-01-01 \
    --eval-start 2023-01-01 \
    --eval-end 2024-01-01 \
    --epochs 30 \
    --output outputs/neutron_model

# 3. 結果確認
cat outputs/neutron_model/metrics.json
```

### 既存モデルで評価のみ

```bash
uv run python scripts/train/train_model.py \
    --reviews data/dataset.csv \
    --train-start 2021-01-01 \
    --train-end 2023-01-01 \
    --eval-start 2023-01-01 \
    --eval-end 2024-01-01 \
    --model outputs/existing_model/irl_model.pt \
    --output outputs/evaluation_results
```

---

## 📊 出力ファイル

```
outputs/model_name/
├── irl_model.pt              # PyTorchモデル重み
├── optimal_threshold.json    # 最適閾値と訓練時性能
│   {
│     "threshold": 0.4909,
│     "precision": 0.457,
│     "recall": 0.983,
│     "f1": 0.624,
│     "method": "f1_maximization_on_train_data"
│   }
├── metrics.json              # 評価メトリクス
│   {
│     "auc_roc": 0.5547,
│     "auc_pr": 0.6118,
│     "f1_score": 0.5824,
│     "positive_count": 60,
│     "negative_count": 75
│   }
├── predictions.csv           # 各レビュアーの予測結果
│   reviewer_email, predicted_prob, true_label, ...
└── eval_trajectories.pkl     # 評価用軌跡データ（分析用）
```

---

## 🔑 重要な設計ポイント

### 1. データリーク防止

- 訓練時: 訓練期間内のみでラベル計算
- 評価時: cutoff 日以前のデータのみで特徴量計算

### 2. 可変長シーケンス対応

- `seq_len=0`で可変長 LSTM
- `pack_padded_sequence`で効率的な処理

### 3. クラス不均衡対策

- Focal Loss で少数クラス重視
- 正例率に応じたパラメータ自動調整

### 4. 閾値決定

- 訓練データで F1 最大化閾値を決定
- 評価時はその閾値を使用（リーク防止）

---

## 📈 モデル性能の目安

| メトリクス | 良好 | 普通     | 要改善 |
| ---------- | ---- | -------- | ------ |
| AUC-ROC    | >0.7 | 0.55-0.7 | <0.55  |
| AUC-PR     | >0.6 | 0.4-0.6  | <0.4   |
| F1 Score   | >0.6 | 0.4-0.6  | <0.4   |

---

_最終更新: 2024 年 12 月_

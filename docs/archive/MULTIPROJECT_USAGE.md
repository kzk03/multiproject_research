# マルチプロジェクトIRL予測システム - 使い方

## 概要

**プロジェクト個別判定 + 複数プロジェクト横断学習**

- 各プロジェクトで継続（レビュー承諾）を**個別に判定**
- 複数プロジェクトでの活動パターンと**相互作用を学習**
- プロジェクト横断的な開発者の行動を捉える

## 論文の実験設定

```
期間: 2021-01-01 ～ 2024-01-01（36ヶ月）
訓練: 2021-01-01 ～ 2023-01-01（24ヶ月）
評価: 2023-01-01 ～ 2024-01-01（12ヶ月）

対象プロジェクト:
- openstack/nova (Compute)
- openstack/neutron (Networking)
- openstack/cinder (Block Storage)
- openstack/keystone (Identity)
- openstack/glance (Image)
```

## クイックスタート（3ステップ）

### ステップ1: データ収集

```bash
# 論文と同じ期間設定で複数プロジェクトのデータを収集
bash scripts/collect_paper_data.sh
```

**出力**: `data/multiproject_paper_data.csv`

このCSVには以下が含まれます：
- `project`: プロジェクト名
- `project_id`: プロジェクトID（= project）
- `label`: レビュー承諾の有無（プロジェクトごとに個別判定）
- `is_cross_project`: クロスプロジェクト活動フラグ
- `reviewer_project_count`: レビュアーの参加プロジェクト数
- その他65種類の特徴量

### ステップ2: IRL形式に変換

```bash
# CSVをIRL形式のJSONに変換
uv run python scripts/convert_to_irl_format.py \
    --input data/multiproject_paper_data.csv \
    --output data/multiproject_irl_data.json
```

**出力**: `data/multiproject_irl_data.json`

### ステップ3: モデル学習と予測

```python
from src.review_predictor.model.irl_predictor import RetentionIRLSystem
import json
from datetime import datetime

# データ読み込み
with open('data/multiproject_irl_data.json', 'r') as f:
    data = json.load(f)

# マルチプロジェクト対応設定
config = {
    'state_dim': 14,  # プロジェクト特徴量を含む
    'action_dim': 5,  # プロジェクト特徴量を含む
    'hidden_dim': 128,
    'learning_rate': 0.0003,
}

# IRLシステム初期化
irl_system = RetentionIRLSystem(config)

# 開発者の継続確率を予測（プロジェクト横断的に）
result = irl_system.predict_continuation_probability(
    developer={
        'developer_id': 'dev@example.com',
        'first_seen': '2023-01-01',
        'changes_authored': 100,
        'changes_reviewed': 50,
        'projects': ['openstack/nova', 'openstack/neutron'],  # 複数プロジェクト
    },
    activity_history=[
        {
            'type': 'review',
            'timestamp': '2024-01-01T00:00:00',
            'project_id': 'openstack/nova',  # このレビューはnovaプロジェクト
            'is_cross_project': True,        # 複数プロジェクトで活動中
            'files_changed': 5,
            'lines_added': 100,
            'lines_deleted': 50,
            'accepted': True,
        },
        {
            'type': 'review',
            'timestamp': '2024-01-05T00:00:00',
            'project_id': 'openstack/neutron',  # このレビューはneutronプロジェクト
            'is_cross_project': True,
            'files_changed': 3,
            'lines_added': 50,
            'lines_deleted': 20,
            'accepted': False,  # neutronでは拒否
        },
    ],
    context_date=datetime.now()
)

print(f"継続確率: {result['continuation_probability']:.2%}")
print(f"理由: {result['reasoning']}")
```

## データの理解

### プロジェクト個別判定とは？

各プロジェクトで開発者の継続を**独立して判定**します。

**例: 開発者Aの場合**

```python
# novaプロジェクトでの活動
nova_activities = [
    {'project_id': 'nova', 'label': 1, 'date': '2023-01'},  # 承諾
    {'project_id': 'nova', 'label': 1, 'date': '2023-02'},  # 承諾
]
→ nova では継続（正例）

# neutronプロジェクトでの活動
neutron_activities = [
    {'project_id': 'neutron', 'label': 0, 'date': '2023-01'},  # 拒否
]
→ neutron では離脱（負例）
```

### 複数プロジェクト横断学習とは？

開発者が複数プロジェクトで活動している場合、**プロジェクト間の相互作用**を学習します。

**学習される特徴量:**
- `project_count`: 参加プロジェクト数
- `project_activity_distribution`: プロジェクト間の活動分散度
- `main_project_contribution_ratio`: メインプロジェクトへの貢献率
- `cross_project_collaboration_score`: プロジェクト横断協力スコア
- `is_cross_project`: クロスプロジェクト活動フラグ

**例: プロジェクト横断パターン**

```python
# 開発者B: nova と neutron で活動
{
    'project_count': 2,
    'project_activity_distribution': 0.3,  # やや偏り（novaが多い）
    'main_project_contribution_ratio': 0.7,  # nova が 70%
    'cross_project_collaboration_score': 0.8,  # 高い横断協力
}
→ novaでの経験がneutronでも活きる可能性が高い
→ 複数プロジェクトで活動することで継続率が上がる
```

## データ構造の詳細

### CSV出力（build_dataset.py）

```csv
change_id,project,project_id,owner_email,reviewer_email,request_time,label,is_cross_project,reviewer_project_count,...
123,openstack/nova,openstack/nova,owner@,reviewer@,2023-01-01,1,True,2,...
124,openstack/neutron,openstack/neutron,owner@,reviewer@,2023-01-02,0,True,2,...
```

**重要なカラム:**
- `project` / `project_id`: プロジェクト名
- `label`: このレビューを承諾したか（1=承諾, 0=拒否）
- `is_cross_project`: 複数プロジェクトで活動中か
- `reviewer_project_count`: レビュアーの参加プロジェクト数

### IRL形式（JSON）

```json
{
  "developers": [
    {
      "developer_id": "dev@example.com",
      "projects": ["openstack/nova", "openstack/neutron"],
      "activity_history": [
        {
          "project_id": "openstack/nova",
          "label": 1,
          "is_cross_project": true,
          ...
        }
      ]
    }
  ]
}
```

## プロジェクトごとの分析

### プロジェクト別の継続率を見る

```python
import pandas as pd

df = pd.read_csv('data/multiproject_paper_data.csv')

# プロジェクトごとの統計
for project in df['project'].unique():
    proj_df = df[df['project'] == project]
    acceptance_rate = proj_df['label'].mean()
    cross_rate = proj_df['is_cross_project'].mean()

    print(f"{project}:")
    print(f"  承諾率: {acceptance_rate:.1%}")
    print(f"  クロスプロジェクト率: {cross_rate:.1%}")
    print()
```

### 開発者のプロジェクト横断パターンを見る

```python
# 複数プロジェクトで活動している開発者を抽出
multi_project_devs = df[df['reviewer_project_count'] > 1]

print(f"マルチプロジェクト開発者: {multi_project_devs['reviewer_email'].nunique()}人")
print(f"全体に占める割合: {multi_project_devs['reviewer_email'].nunique() / df['reviewer_email'].nunique():.1%}")

# マルチプロジェクト開発者の継続率
single_project_rate = df[df['reviewer_project_count'] == 1]['label'].mean()
multi_project_rate = multi_project_devs['label'].mean()

print(f"単一プロジェクト開発者の継続率: {single_project_rate:.1%}")
print(f"マルチプロジェクト開発者の継続率: {multi_project_rate:.1%}")
```

## カスタマイズ

### 対象プロジェクトの変更

`scripts/collect_paper_data.sh` を編集：

```bash
PROJECTS=(
    "openstack/nova"
    "openstack/neutron"
    "your/custom-project"  # 追加
)
```

### 期間の変更

```bash
START_DATE="2022-01-01"
END_DATE="2024-12-31"
```

### レスポンス期間の変更

```bash
# 14日 → 7日に変更
--response-window 7
```

## トラブルシューティング

### データ収集でタイムアウトする

大規模プロジェクトの場合、期間を短くして複数回に分けて収集：

```bash
# 2021年のみ
uv run python scripts/pipeline/build_dataset.py \
    --project openstack/nova \
    --start-date 2021-01-01 \
    --end-date 2022-01-01 \
    --output data/nova_2021.csv

# 2022年のみ
uv run python scripts/pipeline/build_dataset.py \
    --project openstack/nova \
    --start-date 2022-01-01 \
    --end-date 2023-01-01 \
    --output data/nova_2022.csv

# 統合
import pandas as pd
df1 = pd.read_csv('data/nova_2021.csv')
df2 = pd.read_csv('data/nova_2022.csv')
pd.concat([df1, df2]).to_csv('data/nova_full.csv', index=False)
```

### プロジェクト間で不均衡がある

サンプル重み付けを使用：

```python
# 各プロジェクトのサンプル数を均等にする
from sklearn.utils.class_weight import compute_sample_weight

df['sample_weight'] = compute_sample_weight(
    class_weight='balanced',
    y=df['project']
)
```

## まとめ

1. **データ収集**: `bash scripts/collect_paper_data.sh` で一発実行
2. **プロジェクト個別判定**: 各プロジェクトで継続を独立判定（`label`）
3. **横断学習**: 複数プロジェクトでの活動パターンを学習（`is_cross_project`, `project_count`等）
4. **相互作用**: プロジェクト間の影響を捉える（`cross_project_collaboration_score`等）

これにより、**どのプロジェクトで離脱しやすいか**、**複数プロジェクトで活動すると継続率が上がるか**などの分析が可能になります。

# 現在の20 OpenStackプロジェクトサマリー

## データセット情報

- **データソース**: `data/openstack_20proj_2020_2024_feat.csv`
- **総プロジェクト数**: 20プロジェクト
- **総レビュー数**: 147,346件
- **期間**: 2020-2024年
- **総レビュアー数**: 1,235名（ユニーク）
- **総特徴量次元数**: 67次元

---

## プロジェクト一覧（レビュー数順）

### Very Large Projects (>5,000 reviews)

| # | プロジェクト名 | レビュー数 | レビュアー数 | 承諾率 | 説明 |
|---|--------------|-----------|------------|--------|------|
| 1 | **openstack/cinder** | 35,269 | 400 | 63.8% | Block Storage service |
| 2 | **openstack/neutron** | 27,666 | 371 | 71.9% | Networking service |
| 3 | **openstack/nova** | 22,906 | 442 | 56.1% | Compute service (仮想マシン管理) |
| 4 | **openstack/ironic** | 11,369 | 213 | 75.8% | Bare Metal service |
| 5 | **openstack/kolla** | 7,891 | 219 | 72.4% | Container deployment |
| 6 | **openstack/manila** | 5,815 | 164 | 52.3% | Shared File Systems |
| 7 | **openstack/octavia** | 5,333 | 144 | 56.6% | Load Balancing service |

**小計**: 116,249件（全体の78.9%）

---

### Large Projects (1,000-5,000 reviews)

| # | プロジェクト名 | レビュー数 | レビュアー数 | 承諾率 | 説明 |
|---|--------------|-----------|------------|--------|------|
| 8 | **openstack/openstack-helm** | 4,703 | 226 | 60.4% | Helm charts for OpenStack |
| 9 | **openstack/openstack-ansible** | 4,354 | 96 | 80.3% | Ansible playbooks |
| 10 | **openstack/horizon** | 3,429 | 159 | 59.3% | Dashboard (Web UI) |
| 11 | **openstack/swift** | 3,377 | 121 | 71.5% | Object Storage |
| 12 | **openstack/glance** | 3,196 | 172 | 56.3% | Image service |
| 13 | **openstack/keystone** | 2,769 | 167 | 43.5% | Identity service (認証) |
| 14 | **openstack/heat** | 2,286 | 156 | 48.9% | Orchestration service |
| 15 | **openstack/magnum** | 2,220 | 131 | 52.6% | Container Infrastructure |
| 16 | **openstack/designate** | 2,006 | 113 | 70.6% | DNS service |
| 17 | **openstack/barbican** | 1,186 | 73 | 39.2% | Key Management service |

**小計**: 29,526件（全体の20.0%）

---

### Medium Projects (500-1,000 reviews)

| # | プロジェクト名 | レビュー数 | レビュアー数 | 承諾率 | 説明 |
|---|--------------|-----------|------------|--------|------|
| 18 | **openstack/placement** | 815 | 39 | 50.8% | Resource tracking |
| 19 | **openstack/trove** | 549 | 61 | 64.1% | Database as a Service |

**小計**: 1,364件（全体の0.9%）

---

### Small Projects (100-500 reviews)

| # | プロジェクト名 | レビュー数 | レビュアー数 | 承諾率 | 説明 |
|---|--------------|-----------|------------|--------|------|
| 20 | **openstack/zun** | 207 | 38 | 69.1% | Container service |

**小計**: 207件（全体の0.1%）

---

## プロジェクトカテゴリー分析

### コアサービス（最重要）
1. **nova** (Compute) - 22,906件
2. **neutron** (Networking) - 27,666件
3. **cinder** (Block Storage) - 35,269件
4. **keystone** (Identity) - 2,769件
5. **glance** (Image) - 3,196件
6. **swift** (Object Storage) - 3,377件
7. **horizon** (Dashboard) - 3,429件

**小計**: 98,612件（全体の66.9%）

### インフラストラクチャサービス
8. **ironic** (Bare Metal) - 11,369件
9. **heat** (Orchestration) - 2,286件
10. **placement** (Resource tracking) - 815件

**小計**: 14,470件（全体の9.8%）

### ストレージ関連
11. **manila** (Shared File Systems) - 5,815件
12. **trove** (Database) - 549件

**小計**: 6,364件（全体の4.3%）

### コンテナ/デプロイメント
13. **kolla** (Container deployment) - 7,891件
14. **magnum** (Container Infrastructure) - 2,220件
15. **zun** (Container service) - 207件
16. **openstack-helm** (Helm charts) - 4,703件
17. **openstack-ansible** (Ansible) - 4,354件

**小計**: 19,375件（全体の13.1%）

### セキュリティ/ネットワーク
18. **octavia** (Load Balancing) - 5,333件
19. **barbican** (Key Management) - 1,186件
20. **designate** (DNS) - 2,006件

**小計**: 8,525件（全体の5.8%）

---

## 統計サマリー

### レビュー数分布
- **平均**: 7,367件/プロジェクト
- **中央値**: 3,301件/プロジェクト
- **範囲**: 207件（zun）〜 35,269件（cinder）
- **標準偏差**: 9,124件

### レビュアー数分布
- **平均**: 169名/プロジェクト
- **中央値**: 150名/プロジェクト
- **範囲**: 38名（zun）〜 442名（nova）
- **標準偏差**: 103名

### 承諾率分布
- **平均**: 60.6%
- **中央値**: 59.8%
- **範囲**: 39.2%（barbican）〜 80.3%（openstack-ansible）
- **標準偏差**: 11.2%

---

## 特徴量構成

### 状態特徴量（State Features）: 14次元
1. プロジェクト関連特徴量（マルチプロジェクトの場合追加）
2. レビュアー経験値
3. 過去承諾率
4. 活動度
5. その他のレビュアー属性

### 行動特徴量（Action Features）: 5次元
1. レビュー依頼の特性
2. コードの複雑さ
3. プロジェクト固有の特徴量（マルチプロジェクトの場合追加）
4. その他のレビュー属性

---

## データ品質評価

### 十分なデータ量のプロジェクト
- **最低100レビュー以上**: 20/20プロジェクト（100%）
- **最低500レビュー以上**: 18/20プロジェクト（90.0%）
- **最低1,000レビュー以上**: 17/20プロジェクト（85.0%）
- **最低5,000レビュー以上**: 7/20プロジェクト（35.0%）

### 予測精度の期待値
- **Very Large (>5K)**: F1 = 0.70-0.75（高精度）
- **Large (1K-5K)**: F1 = 0.65-0.72（中〜高精度）
- **Medium (500-1K)**: F1 = 0.60-0.68（中精度）
- **Small (<500)**: F1 = 0.55-0.65（低〜中精度、データ不足の影響）

---

## マルチプロジェクトモデルの有効性

### データ共有の効果
- **Top 7プロジェクト**: 全データの78.9%をカバー
- **Top 10プロジェクト**: 全データの87.9%をカバー
- **Top 17プロジェクト**: 全データの98.9%をカバー

### クロスプロジェクト学習
大規模プロジェクト（cinder, neutron, nova）の学習が、小規模プロジェクト（zun, trove）の予測精度向上に寄与する可能性が高い。

特に：
- **同カテゴリ内**: コンテナ関連（kolla, magnum, zun）など、類似プロジェクト間での知識共有
- **コアサービス**: 多くのレビュアーが複数のコアサービスに参加しているため、レビュアー特性の共有学習が有効

---

## 次のステップ: 全OpenStackプロジェクト調査

### 調査項目
1. OpenStack公式の全プロジェクト数
2. 各プロジェクトのGerritレビュー数
3. データ取得可能性
4. 予測可能な最小データ量の定義

### 期待される拡張
- 現在の20プロジェクト → **全プロジェクト（推定: 50-100+プロジェクト）**
- レビュー数: 147,346件 → **推定: 500,000-1,000,000+件**
- カバー期間の拡大: 2020-2024 → 2010-2024（OpenStack開始時点から）


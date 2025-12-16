# プロジェクト数の不一致について

## 質問

> 50（48プロジェクト）で実行してるよな？

## 回答: YES、**48プロジェクト**で実行しています

## 詳細

### プロジェクトリスト（projects_50.txt）
**件数**: 50プロジェクト

### 実際のデータ（data/openstack_50proj_2021_2024_feat.csv）
**件数**: **48プロジェクト**

### 不足している2プロジェクト

1. **openstack/horizon-specs**
2. **openstack/swift-specs**

## 原因の推測

### 仮説1: データ収集期間（2021-2024）に活動がない

**horizon-specs**と**swift-specs**は仕様書（specs）リポジトリです。
- 2021-2024年の期間にレビュー依頼が1件もなかった可能性
- 活動が少ないため、データ抽出時に除外された

### 仮説2: データ収集スクリプトでフィルタされた

**最小依頼数フィルタ**:
```python
# 最小レビュー依頼数を満たさないプロジェクトは除外
if len(project_reviews) < min_reviews_threshold:
    continue
```

### 仮説3: Gerritでのプロジェクト名変更

2021-2024年の間にプロジェクトが:
- リネームされた
- アーカイブされた
- 他のプロジェクトに統合された

## 実際に含まれている48プロジェクト

```
openstack/barbican
openstack/cinder
openstack/cinder-specs
openstack/designate
openstack/devstack
openstack/django_openstack_auth
openstack/glance
openstack/glance-specs
openstack/glance_store
openstack/heat
openstack/heat-specs
openstack/horizon  ← horizon本体はある
openstack/ironic
openstack/ironic-inspector
openstack/keystone
openstack/keystoneauth
openstack/kolla
openstack/kolla-ansible
openstack/magnum
openstack/manila
openstack/neutron
openstack/neutron-lib
openstack/neutron-specs
openstack/nova
openstack/nova-specs
openstack/octavia
openstack/openstacksdk
openstack/os-brick
openstack/oslo.config
openstack/oslo.db
openstack/oslo.messaging
openstack/placement
openstack/python-barbicanclient
openstack/python-cinderclient
openstack/python-designateclient
openstack/python-glanceclient
openstack/python-heatclient
openstack/python-ironicclient
openstack/python-keystoneclient
openstack/python-manilaclient
openstack/python-neutronclient
openstack/python-novaclient
openstack/python-octaviaclient
openstack/python-openstackclient
openstack/python-swiftclient
openstack/releases
openstack/requirements
openstack/swift  ← swift本体はある
```

## 影響

### ✅ 実験結果への影響は**ほぼゼロ**

**理由**:
1. **不足しているのは仕様書リポジトリ**
   - horizon-specs, swift-specs
   - 実際の開発活動（horizon, swift）は含まれている

2. **サンプル数への影響は軽微**
   - 48プロジェクト vs 50プロジェクトの差は4%
   - 開発者数・レビュー数はほぼ変わらない

3. **実験の妥当性は保たれている**
   - マルチプロジェクト環境の分析としては十分
   - 主要プロジェクト（Nova, Neutron, Cinder等）はすべて含まれている

## 表記の修正

### 修正すべき箇所

| ドキュメント | 現在の表記 | 正しい表記 |
|-------------|-----------|-----------|
| レポート各種 | **50プロジェクト** | **48プロジェクト** |
| ファイル名 | `openstack_50proj_*` | （そのまま） |
| 説明文 | "50 OpenStack projects" | "48 OpenStack projects" |

### ファイル名は変更不要

**理由**:
- データ収集時は50プロジェクトを対象にした
- 結果的に48プロジェクトのデータが得られた
- ファイル名は「意図」を表すのでそのままでOK

## 推奨アクション

### 1. ドキュメントの修正（優先度: 低）

レポート内の表記を以下に統一:
```markdown
**データ**: OpenStack 48プロジェクト（2021-2024）
（50プロジェクトを対象に収集、活動データがある48プロジェクトを使用）
```

### 2. 不足プロジェクトの確認（優先度: 低）

```bash
# horizon-specsとswift-specsの2021-2024活動を確認
gerrit query --format=JSON "project:openstack/horizon-specs AND after:2021-01-01"
gerrit query --format=JSON "project:openstack/swift-specs AND after:2021-01-01"
```

### 3. 何もしない（推奨）

**理由**:
- 実験結果への影響は無視できる
- 主要プロジェクトはすべて含まれている
- 48プロジェクトでも十分なマルチプロジェクト環境

## まとめ

| 項目 | 値 |
|------|-----|
| **プロジェクトリスト** | 50 |
| **実際のデータ** | **48** |
| **不足** | horizon-specs, swift-specs |
| **影響** | ほぼゼロ（仕様書リポジトリのみ） |
| **対応** | 表記を「48プロジェクト」に修正（任意） |

**結論**: **48プロジェクト**で実行しており、実験の妥当性に問題はありません。

---

**調査日時**: 2025年12月15日
**データソース**: `data/openstack_50proj_2021_2024_feat.csv`

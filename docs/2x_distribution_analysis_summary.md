# 予測成功/失敗開発者の分布分析レポート

**分析対象**: 2x OSモデル（6-9m）の183開発者
**分析日**: 2025-12-15

---

## エグゼクティブサマリー

### 予測結果
- **予測成功**: 145名 (79.2%)
- **予測失敗**: 38名 (20.8%)

### 主要発見

1. **レビュー回数の決定的な差**
   - 予測成功: 訓練期間平均**54.1件**、評価期間平均**42.2件**
   - 予測失敗: 訓練期間平均**15.4件**、評価期間平均**12.6件**
   - **差**: 予測成功者は**3.5倍多い**レビュー経験

2. **企業アカウントの精度差**
   - **Red Hat**: 87.8%精度（41名中36名成功）← 最高精度
   - **Personal (Gmail等)**: 81.5%精度（54名中44名成功）
   - **Other Company**: 76.1%精度（71名中54名成功）
   - **Dell, CERN**: 0%精度（全員失敗）

3. **予測失敗の特徴**
   - 訓練期間レビュー数中央値: **5件** vs 成功者24件
   - 総レビュー数中央値: **0件** vs 成功者11件
   - プロジェクト数: 平均**2.1個** vs 成功者4.8個

---

## 1. レビュー回数分布分析

### 統計比較

| 指標 | 予測成功 | 予測失敗 | 差 |
|------|---------|---------|-----|
| **訓練期間レビュー数（平均）** | 54.1件 | 15.4件 | **+252%** |
| **訓練期間レビュー数（中央値）** | 24件 | 5件 | **+380%** |
| **評価期間レビュー数（平均）** | 42.2件 | 12.6件 | **+235%** |
| **評価期間レビュー数（中央値）** | 20件 | 4件 | **+400%** |
| **総レビュー数（平均）** | 35.9件 | 8.8件 | **+308%** |
| **総レビュー数（中央値）** | 11件 | 0件 | **∞** |

### 重要な洞察

**予測成功する開発者**:
- 訓練期間で**24件以上**のレビュー経験（中央値）
- 評価期間でも**20件以上**の継続活動
- **一貫した活動パターン**（訓練→評価で継続）

**予測失敗する開発者**:
- 訓練期間で**5件以下**のレビュー経験（中央値）
- 評価期間で**4件以下**の低活動
- **総レビュー数が0件**（中央値）← 新規参加者が多い

**可視化**: [review_count_distribution.png](../outputs/analysis_data/2x_distribution/review_count_distribution.png)

---

## 2. メールドメイン（企業アカウント）分布分析

### ドメインタイプ別統計

| ドメインタイプ | 総数 | 成功 | 失敗 | 精度 |
|--------------|------|------|------|------|
| **Red Hat** | 41 | 36 | 5 | **87.8%** ← 最高 |
| **StackHPC** | 6 | 5 | 1 | 83.3% |
| **Personal (Gmail等)** | 54 | 44 | 10 | 81.5% |
| **Other Company** | 71 | 54 | 17 | 76.1% |
| **Mirantis** | 2 | 2 | 0 | 100.0% |
| **NVIDIA** | 1 | 1 | 0 | 100.0% |
| **Academic** | 1 | 1 | 0 | 100.0% |
| **Canonical** | 2 | 1 | 1 | 50.0% |
| **Huawei** | 2 | 1 | 1 | 50.0% |
| **Dell** | 2 | 0 | 2 | **0.0%** ← 全員失敗 |
| **CERN** | 1 | 0 | 1 | **0.0%** ← 全員失敗 |

### Top 15 ドメイン詳細

| 順位 | ドメイン | タイプ | 総数 | 精度 |
|------|---------|--------|------|------|
| 1 | gmail.com | Personal | 51 | 82.4% |
| 2 | redhat.com | Red Hat | 41 | **87.8%** |
| 3 | stackhpc.com | StackHPC | 6 | 83.3% |
| 4 | tristero.net | Other | 3 | 100.0% |
| 5 | inspur.com | Other | 3 | 66.7% |
| 6 | zte.com.cn | Other | 3 | 66.7% |
| 7 | yovole.com | Other | 3 | 100.0% |
| 8 | canonical.com | Canonical | 2 | 50.0% |
| 9 | mirantis.com | Mirantis | 2 | 100.0% |
| 10 | dell.com | Dell | 2 | **0.0%** |
| 11 | qq.com | Other | 2 | 50.0% |
| 12 | yadro.com | Other | 2 | 100.0% |
| 13 | huawei.com | Huawei | 2 | 50.0% |
| 14 | 163.com | Other | 2 | 100.0% |
| 15 | binero.com | Other | 1 | 100.0% |

### 企業別の特徴

**Red Hat（87.8%精度、41名）**:
- OpenStackの**主要貢献企業**
- 開発者が**多数プロジェクトに参加**（平均6.5プロジェクト）
- **一貫した活動パターン**（平均活動間隔6日）
- 代表的な成功例:
  - stephenfin@redhat.com: 13プロジェクト、164件訓練レビュー
  - ralonsoh@redhat.com: 11プロジェクト、270件訓練レビュー
  - skaplons@redhat.com: 8プロジェクト、293件訓練レビュー

**Personal (Gmail等, 81.5%精度、54名）**:
- **個人開発者**が中心
- プロジェクト数のばらつきが大きい（1-12プロジェクト）
- 成功者は**Expert開発者**が多い（9+プロジェクト）
- 代表的な成功例:
  - gibizer@gmail.com: 12プロジェクト、211件訓練レビュー
  - sean.mcginnis@gmail.com: 11プロジェクト、181件訓練レビュー
  - katonalala@gmail.com: 9プロジェクト、217件訓練レビュー

**Other Company（76.1%精度、71名）**:
- **中小企業・国内企業**が中心（中国企業が多い）
- プロジェクト数が少ない（平均3.2プロジェクト）
- 活動が散発的（平均活動間隔9日）
- 失敗例:
  - zte.com.cn, inspur.com: 1-4プロジェクト、低頻度

**Dell（0%精度、2名全員失敗）**:
- 訓練期間レビュー数: 3件、8件
- プロジェクト数: 2個
- 予測確率: 0.47（低い継続確率）
- **理由**: 活動が散発的で予測困難

**CERN（0%精度、1名失敗）**:
- spyridon.trigazis@cern.ch
- 訓練期間: 22件、プロジェクト数2個
- 評価期間: 14件（実際は活動継続）
- 予測確率: 0.76（継続予測だが不正解）
- **理由**: 承諾率62.5%と中途半端で、予測が外れた

**可視化**: [domain_distribution.png](../outputs/analysis_data/2x_distribution/domain_distribution.png)

---

## 3. 予測成功開発者の詳細プロファイル

### Top 10 予測成功開発者（プロジェクト数順）

| 順位 | メールアドレス | ドメイン | プロジェクト数 | 訓練レビュー | 評価レビュー | 予測確率 |
|------|--------------|---------|-------------|-------------|-------------|---------|
| 1 | elod.illes@est.tech | Other | **18** | 371 | 472 | 0.925 |
| 2 | likui@yovole.com | Other | **16** | 67 | 67 | 0.858 |
| 3 | gmaan@ghanshyammann.com | Other | **15** | 105 | 85 | 0.899 |
| 4 | stephenfin@redhat.com | Red Hat | **13** | 164 | 86 | 0.891 |
| 5 | gibizer@gmail.com | Gmail | **12** | 211 | 29 | 0.907 |
| 6 | kajinamit@oss.nttdata.com | Other | **11** | 48 | 99 | 0.810 |
| 7 | sean.mcginnis@gmail.com | Gmail | **11** | 181 | 19 | 0.905 |
| 8 | ralonsoh@redhat.com | Red Hat | **11** | 270 | 182 | 0.916 |
| 9 | frickler@offenerstapel.de | Other | **10** | 65 | 167 | 0.880 |
| 10 | rosmaita.fossdev@gmail.com | Gmail | **9** | 177 | 52 | 0.905 |

**共通特徴**:
- **9+プロジェクト**でのマルチプロジェクト活動
- 訓練期間で**48件以上**のレビュー経験
- 予測確率**0.80以上**（高い継続確率）
- 企業・個人問わず高精度

---

## 4. 予測失敗開発者の詳細プロファイル

### Top 10 予測失敗開発者（プロジェクト数順）

| 順位 | メールアドレス | ドメイン | プロジェクト数 | 訓練レビュー | 評価レビュー | 予測確率 | 理由 |
|------|--------------|---------|-------------|-------------|-------------|---------|------|
| 1 | radek@piliszek.it | Other | **8** | 284 | 1 | 0.915 | FP: 高活動だが評価期間で離脱 |
| 2 | nurmatov.mamatisa@huawei.com | Huawei | 4 | 35 | 1 | 0.864 | FP: 中活動だが離脱 |
| 3 | corey.bryant@canonical.com | Canonical | 3 | 4 | 1 | 0.556 | FP: 低活動、散発的 |
| 4 | galkindmitrii@gmail.com | Gmail | 3 | 6 | 2 | 0.588 | FN: 新規参加者 |
| 5 | caiquemellosbo@gmail.com | Gmail | 3 | 7 | 5 | 0.499 | FN: 新規参加者 |
| 6 | anlin.kong@gmail.com | Gmail | 2 | 11 | 13 | 0.785 | FP: 中活動だが継続 |
| 7 | oschwart@redhat.com | Red Hat | 2 | 3 | 11 | 0.476 | FN: 低活動から復活 |
| 8 | felix.huettner@stackit.cloud | Other | 2 | 7 | 1 | 0.492 | FN: 低活動、散発的 |
| 9 | cgoncalves@redhat.com | Red Hat | 2 | 29 | 16 | 0.633 | FP: 承諾率0%で離脱 |
| 10 | vhariria@redhat.com | Red Hat | 2 | 5 | 3 | 0.483 | FN: 低活動から復活 |

### 失敗パターン分類

**False Positive（活動継続と予測したが離脱）**:
- radek@piliszek.it: **284件**の豊富な訓練経験があったが、評価期間で**1件のみ**
- nurmatov.mamatisa@huawei.com: 承諾率64.7%と中程度だったが離脱
- anlin.kong@gmail.com: 承諾率80%と高かったが、訓練期間が少なく不安定

**False Negative（離脱と予測したが継続）**:
- oschwart@redhat.com: 訓練期間**3件**のみで離脱予測、評価期間で**11件**に復活
- felix.huettner@stackit.cloud: 訓練期間**7件**、散発的だったが継続
- caiquemellosbo@gmail.com: 訓練期間**7件**、新規参加者で不安定

### 失敗の主な原因

1. **訓練期間の活動が少ない**（中央値5件 vs 成功24件）
2. **プロジェクト数が少ない**（平均2.1個 vs 成功4.8個）
3. **承諾率が極端**（0%または50%前後で不安定）
4. **活動の季節性・一時的変動**（休暇、プロジェクト変更など）

---

## 5. 特徴量分布の詳細比較

### 主要特徴量の分布差

| 特徴量 | 予測成功（平均） | 予測失敗（平均） | 差 |
|--------|---------------|---------------|-----|
| **project_count** | 4.8 | 2.1 | +129% |
| **recent_activity_frequency** | 0.68 | 0.15 | +353% |
| **recent_acceptance_rate** | 0.59 | 0.26 | +127% |
| **experience_days** | 80.5日 | 64.4日 | +25% |
| **avg_activity_gap** | 6.1日 | 15.7日 | -61% |
| **cross_project_collaboration_score** | 0.91 | 0.54 | +69% |

**可視化**: [feature_distributions.png](../outputs/analysis_data/2x_distribution/feature_distributions.png)

---

## 6. 実用的な示唆

### レビュアー推薦システムの改善

**高精度で推薦できる開発者**:
- Red Hat開発者（87.8%精度）
- Gmail個人開発者でExpert（9+プロジェクト、81.5%精度）
- 訓練期間レビュー数24件以上

**推薦アルゴリズム**:
```python
def is_reliable_reviewer(developer):
    return (
        developer.history_reviews >= 24 and
        developer.project_count >= 4 and
        developer.recent_acceptance_rate >= 0.5
    )
```

### 企業別の育成戦略

**Red Hat（高精度企業）の成功要因**:
- マルチプロジェクト参加の奨励
- 定期的なレビュー活動の文化
- 一貫した品質基準

**Other Company（中精度企業）の改善策**:
- プロジェクト数を増やす（3→5+）
- レビュー頻度を上げる（週1→週2）
- 承諾率の安定化（品質向上）

**Dell, CERN（低精度企業）の課題**:
- 活動の一貫性確保
- プロジェクト参加数の増加
- 長期コミットメントの促進

---

## 7. 生成ファイル一覧

### データファイル
- [review_count_stats.csv](../outputs/analysis_data/2x_distribution/review_count_stats.csv) - レビュー回数統計
- [domain_type_stats.csv](../outputs/analysis_data/2x_distribution/domain_type_stats.csv) - ドメインタイプ別統計
- [domain_detail_stats.csv](../outputs/analysis_data/2x_distribution/domain_detail_stats.csv) - Top 15ドメイン詳細
- [correct_developers.csv](../outputs/analysis_data/2x_distribution/correct_developers.csv) - 予測成功開発者145名
- [incorrect_developers.csv](../outputs/analysis_data/2x_distribution/incorrect_developers.csv) - 予測失敗開発者38名
- [all_developers.csv](../outputs/analysis_data/2x_distribution/all_developers.csv) - 全開発者183名

### 可視化
- [review_count_distribution.png](../outputs/analysis_data/2x_distribution/review_count_distribution.png) - レビュー回数分布（ヒストグラム+ボックスプロット）
- [domain_distribution.png](../outputs/analysis_data/2x_distribution/domain_distribution.png) - ドメイン分布（棒グラフ+円グラフ）
- [feature_distributions.png](../outputs/analysis_data/2x_distribution/feature_distributions.png) - 特徴量分布比較

---

## 8. 結論

### 主要成果

1. **レビュー回数が最重要指標**: 予測成功者は**3.5倍多い**レビュー経験（54件 vs 15件）
2. **Red Hatが最高精度**: 87.8%の予測精度（主要貢献企業の優位性）
3. **個人開発者も高精度**: Gmail等で81.5%（Expertレベルなら十分予測可能）
4. **失敗パターンの明確化**: 訓練期間5件以下、プロジェクト数2個以下が危険信号

### 実用的価値

- **企業別ベンチマーク**: Red Hat（87.8%）をターゲットに他企業も改善可能
- **個人開発者の育成**: Gmail開発者でもマルチプロジェクト参加で高精度
- **推薦システム改善**: レビュー数24件、プロジェクト数4個を閾値に設定

### 今後の課題

1. **Dell, CERNの低精度改善**: 活動の一貫性確保策を検討
2. **False Positive削減**: 季節性・一時的変動への対応（10.9% → 5%目標）
3. **新規参加者の予測**: 訓練期間が短い開発者への対策

---

**Report Generated**: 2025-12-15
**Total Developers Analyzed**: 183
**Success Rate**: 79.2%

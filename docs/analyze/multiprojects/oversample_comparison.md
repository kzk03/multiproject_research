# マルチプロジェクト継続予測：負例オーバーサンプリング比較 (x2 vs x3)

## サマリ

- 実験: `scripts/train/train_cross_temporal_multiproject.py` を負例オーバーサンプリング係数 2 (`os2x`), 3 (`os3x`) で全 10 パターン実行。
- 評価分布: 評価データは全パターンでほぼ均衡 (pos≈0.42–0.52)。サンプリングは訓練時のみ作用。
- AUC-ROC: x3 は初期〜中期窓を中心に 1–3pt 程度底上げ、逆に 0-3→3-6 は僅かに低下。
- 実際の予測スコア: `results/cross_temporal_openstack_multiproject_os{2,3}x/train_*/eval_*/predictions.csv` に含まれる (列例: reviewer, y_true, y_pred, score)。

## メトリクス比較（AUC-ROC）

| train→eval  | x2     | x3     | 備考      |
| ----------- | ------ | ------ | --------- |
| 0-3m→0-3m   | 0.6661 | 0.6865 | x3 +2.0pt |
| 0-3m→3-6m   | 0.7872 | 0.7679 | x3 -1.9pt |
| 0-3m→6-9m   | 0.7185 | 0.7272 | x3 +0.9pt |
| 0-3m→9-12m  | 0.7000 | 0.6938 | x3 -0.6pt |
| 3-6m→3-6m   | 0.7523 | 0.7494 | x3 -0.3pt |
| 3-6m→6-9m   | 0.6444 | 0.6756 | x3 +3.1pt |
| 3-6m→9-12m  | 0.6553 | 0.6671 | x3 +1.2pt |
| 6-9m→6-9m   | 0.6758 | 0.7085 | x3 +3.3pt |
| 6-9m→9-12m  | 0.6678 | 0.6652 | x3 -0.3pt |
| 9-12m→9-12m | 0.6930 | 0.7053 | x3 +1.2pt |

## クラス割合（学習 vs 予測）

- 学習: サンプリング後の負例を含む。例 `0-3m` 訓練では pos 125 / neg 180 (pos 率 41.0%)。
- 予測: 生データ分布。例 `0-3m→0-3m` 評価は pos 51 / neg 48 (pos 率 51.5%)。
- 詳細表は前回集計の通り (pos 率は評価で 0.42–0.52 の範囲で安定)。

## 実際の予測結果の場所

- x2: `results/cross_temporal_openstack_multiproject_os2x/train_*m/eval_*m/predictions.csv`
- x3: `results/cross_temporal_openstack_multiproject_os3x/train_*m/eval_*m/predictions.csv`
  - 各 CSV に予測スコア (`score` など) と真値 (`y_true`) が含まれる。レビューア ID や期間ごとに確認可能。

## 所感と次アクション案

- x3 は中期以降 (3-6→6-9, 6-9→6-9, 9-12→9-12) で顕著に底上げ。初期 → 中期 (0-3→3-6) では軽微に悪化。
- 使い分け案: 窓ごとに x2/x3 のベストを採用するメタ選択もあり。
- 追加検証: 閾値校正 (Platt/Isotonic) や α/γ 固定の再学習で ROC をさらに押し上げられる可能性。

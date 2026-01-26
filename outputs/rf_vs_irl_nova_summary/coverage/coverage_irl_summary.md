# IRL/RF カバレッジ結果まとめ (rf_vs_irl_nova_summary)

- IRL: coverage_irl_only_all.csv (threshold=0.5 で全件 pred=0)
- RF: coverage_rf_only_0-3m_to_6-9m.csv (split=0-3m_to_6-9m, 4 件)
- ユーザ数: 24, IRL 行数: 60, RF 行数: 4
- グループ定義: Heavy=eval_req>=50 / Light=それ未満; Low <0.1, Mid <0.3, High >=0.3

## 開発者別サマリ (全パターン、モデル明記)

### melwittt@gmail.com

- グループ: Heavy-Mid
- eval 合計: 178 件 (受理 38, 却下 140, 受理率 0.213)
- 履歴: req 2315 件, 平均受理率 0.203
- 予測結果一覧 (model 列で IRL / RF を区別):
  |model|train|eval|split|prob|pred|true|hist_req|hist_rate|eval_req|eval_acc|eval_rej|
  |---|---|---|---|---|---|---|---|---|---|---|---|
  |IRL|0-3m|0-3m|-|0.463|0|1|463|0.203023758099352|45|10|35|
  |IRL|0-3m|3-6m|-|0.463|0|1|463|0.203023758099352|31|8|23|
  |IRL|0-3m|6-9m|-|0.463|0|1|463|0.203023758099352|28|4|24|
  |IRL|0-3m|9-12m|-|0.463|0|1|463|0.203023758099352|37|8|29|
  |IRL|9-12m|9-12m|-|0.478|0|1|463|0.203023758099352|37|8|29|

### ratailor@redhat.com

- グループ: Heavy-Low
- eval 合計: 157 件 (受理 5, 却下 152, 受理率 0.032)
- 履歴: req 420 件, 平均受理率 0.036
- 予測結果一覧 (model 列で IRL / RF を区別):
  |model|train|eval|split|prob|pred|true|hist_req|hist_rate|eval_req|eval_acc|eval_rej|
  |---|---|---|---|---|---|---|---|---|---|---|---|
  |IRL|0-3m|0-3m|-|0.460|0|1|84|0.0357142857142857|63|1|62|
  |IRL|0-3m|3-6m|-|0.460|0|0|84|0.0357142857142857|28|0|28|
  |IRL|0-3m|6-9m|-|0.460|0|0|84|0.0357142857142857|4|0|4|
  |IRL|0-3m|9-12m|-|0.460|0|1|84|0.0357142857142857|31|2|29|
  |IRL|9-12m|9-12m|-|0.477|0|1|84|0.0357142857142857|31|2|29|

### auniyal@redhat.com

- グループ: Heavy-Low
- eval 合計: 148 件 (受理 9, 却下 139, 受理率 0.061)
- 履歴: req 60 件, 平均受理率 0.200
- 予測結果一覧 (model 列で IRL / RF を区別):
  |model|train|eval|split|prob|pred|true|hist_req|hist_rate|eval_req|eval_acc|eval_rej|
  |---|---|---|---|---|---|---|---|---|---|---|---|
  |IRL|0-3m|0-3m|-|0.476|0|1|15|0.2|32|1|31|
  |IRL|0-3m|3-6m|-|0.476|0|1|15|0.2|49|4|45|
  |IRL|0-3m|6-9m|-|0.476|0|1|15|0.2|48|3|45|
  |IRL|0-3m|9-12m|-|0.476|0|1|15|0.2|19|1|18|

### mlnx-openstack-ci@dev.mellanox.co.il

- グループ: Heavy-Low
- eval 合計: 118 件 (受理 1, 却下 117, 受理率 0.008)
- 履歴: req 2035 件, 平均受理率 0.000
- 予測結果一覧 (model 列で IRL / RF を区別):
  |model|train|eval|split|prob|pred|true|hist_req|hist_rate|eval_req|eval_acc|eval_rej|
  |---|---|---|---|---|---|---|---|---|---|---|---|
  |IRL|0-3m|0-3m|-|0.463|0|0|407|0.0|49|0|49|
  |IRL|0-3m|3-6m|-|0.463|0|1|407|0.0|34|1|33|
  |IRL|0-3m|6-9m|-|0.463|0|0|407|0.0|31|0|31|
  |IRL|0-3m|9-12m|-|0.463|0|0|407|0.0|2|0|2|
  |IRL|9-12m|9-12m|-|0.477|0|0|407|0.0|2|0|2|

### emc.scaleio.ci@emc.com

- グループ: Heavy-Low
- eval 合計: 103 件 (受理 1, 却下 102, 受理率 0.010)
- 履歴: req 2182 件, 平均受理率 0.004
- 予測結果一覧 (model 列で IRL / RF を区別):
  |model|train|eval|split|prob|pred|true|hist_req|hist_rate|eval_req|eval_acc|eval_rej|
  |---|---|---|---|---|---|---|---|---|---|---|---|
  |IRL|0-3m|0-3m|-|0.468|0|1|1091|0.0036663611365719|88|1|87|
  |IRL|0-3m|3-6m|-|0.468|0|0|1091|0.0036663611365719|15|0|15|

### takanattie@gmail.com

- グループ: Heavy-Low
- eval 合計: 95 件 (受理 1, 却下 94, 受理率 0.011)
- 履歴: req 2255 件, 平均受理率 0.053
- 予測結果一覧 (model 列で IRL / RF を区別):
  |model|train|eval|split|prob|pred|true|hist_req|hist_rate|eval_req|eval_acc|eval_rej|
  |---|---|---|---|---|---|---|---|---|---|---|---|
  |IRL|0-3m|0-3m|-|0.463|0|0|451|0.0532150776053215|36|0|36|
  |IRL|0-3m|3-6m|-|0.463|0|1|451|0.0532150776053215|29|1|28|
  |IRL|0-3m|6-9m|-|0.463|0|0|451|0.0532150776053215|10|0|10|
  |IRL|0-3m|9-12m|-|0.463|0|0|451|0.0532150776053215|10|0|10|
  |IRL|9-12m|9-12m|-|0.478|0|0|451|0.0532150776053215|10|0|10|

### notartom@gmail.com

- グループ: Heavy-Mid
- eval 合計: 55 件 (受理 16, 却下 39, 受理率 0.291)
- 履歴: req 530 件, 平均受理率 0.132
- 予測結果一覧 (model 列で IRL / RF を区別):
  |model|train|eval|split|prob|pred|true|hist_req|hist_rate|eval_req|eval_acc|eval_rej|
  |---|---|---|---|---|---|---|---|---|---|---|---|
  |IRL|0-3m|0-3m|-|0.458|0|1|106|0.1320754716981132|25|7|18|
  |IRL|0-3m|3-6m|-|0.458|0|1|106|0.1320754716981132|9|2|7|
  |IRL|0-3m|6-9m|-|0.458|0|1|106|0.1320754716981132|5|1|4|
  |IRL|0-3m|9-12m|-|0.458|0|1|106|0.1320754716981132|8|3|5|
  |IRL|9-12m|9-12m|-|0.475|0|1|106|0.1320754716981132|8|3|5|

### dms@danplanet.com

- グループ: Light-Mid
- eval 合計: 27 件 (受理 5, 却下 22, 受理率 0.185)
- 履歴: req 128 件, 平均受理率 0.227
- 予測結果一覧 (model 列で IRL / RF を区別):
  |model|train|eval|split|prob|pred|true|hist_req|hist_rate|eval_req|eval_acc|eval_rej|
  |---|---|---|---|---|---|---|---|---|---|---|---|
  |IRL|9-12m|9-12m|-|0.477|0|1|128|0.2265625|27|5|22|

### jay@jvf.cc

- グループ: Light-Mid
- eval 合計: 22 件 (受理 3, 却下 19, 受理率 0.136)
- 履歴: req 14 件, 平均受理率 0.714
- 予測結果一覧 (model 列で IRL / RF を区別):
  |model|train|eval|split|prob|pred|true|hist_req|hist_rate|eval_req|eval_acc|eval_rej|
  |---|---|---|---|---|---|---|---|---|---|---|---|
  |IRL|6-9m|6-9m|-|0.478|0|0|7|0.7142857142857143|6|0|6|
  |IRL|6-9m|9-12m|-|0.478|0|1|7|0.7142857142857143|16|3|13|

### openstack@lightbitslabs.com

- グループ: Light-Low
- eval 合計: 17 件 (受理 0, 却下 17, 受理率 0.000)
- 履歴: req 563 件, 平均受理率 0.005
- 予測結果一覧 (model 列で IRL / RF を区別):
  |model|train|eval|split|prob|pred|true|hist_req|hist_rate|eval_req|eval_acc|eval_rej|
  |---|---|---|---|---|---|---|---|---|---|---|---|
  |IRL|9-12m|9-12m|-|0.480|0|0|563|0.0053285968028419|17|0|17|
  |RF|-|-|0-3m_to_6-9m|0.565|1|0|-|-|-|-|-|

### gmaan@ghanshyammann.com

- グループ: Light-Low
- eval 合計: 14 件 (受理 1, 却下 13, 受理率 0.071)
- 履歴: req 164 件, 平均受理率 0.311
- 予測結果一覧 (model 列で IRL / RF を区別):
  |model|train|eval|split|prob|pred|true|hist_req|hist_rate|eval_req|eval_acc|eval_rej|
  |---|---|---|---|---|---|---|---|---|---|---|---|
  |IRL|9-12m|9-12m|-|0.479|0|1|164|0.3109756097560975|14|1|13|

### aleksey.stupnikov@gmail.com

- グループ: Light-High
- eval 合計: 9 件 (受理 3, 却下 6, 受理率 0.333)
- 履歴: req 96 件, 平均受理率 0.083
- 予測結果一覧 (model 列で IRL / RF を区別):
  |model|train|eval|split|prob|pred|true|hist_req|hist_rate|eval_req|eval_acc|eval_rej|
  |---|---|---|---|---|---|---|---|---|---|---|---|
  |IRL|9-12m|9-12m|-|0.478|0|1|96|0.0833333333333333|9|3|6|
  |RF|-|-|0-3m_to_6-9m|0.593|1|0|-|-|-|-|-|

### christian.rohmann@inovex.de

- グループ: Light-Mid
- eval 合計: 7 件 (受理 1, 却下 6, 受理率 0.143)
- 履歴: req 3 件, 平均受理率 0.000
- 予測結果一覧 (model 列で IRL / RF を区別):
  |model|train|eval|split|prob|pred|true|hist_req|hist_rate|eval_req|eval_acc|eval_rej|
  |---|---|---|---|---|---|---|---|---|---|---|---|
  |IRL|9-12m|9-12m|-|0.475|0|1|3|0.0|7|1|6|

### lyarwood@redhat.com

- グループ: Light-Low
- eval 合計: 7 件 (受理 0, 却下 7, 受理率 0.000)
- 履歴: req 1144 件, 平均受理率 0.199
- 予測結果一覧 (model 列で IRL / RF を区別):
  |model|train|eval|split|prob|pred|true|hist_req|hist_rate|eval_req|eval_acc|eval_rej|
  |---|---|---|---|---|---|---|---|---|---|---|---|
  |IRL|0-3m|0-3m|-|0.463|0|0|572|0.1993006993006993|5|0|5|
  |IRL|0-3m|3-6m|-|0.463|0|0|572|0.1993006993006993|2|0|2|

### juliaashleykreger@gmail.com

- グループ: Light-Low
- eval 合計: 6 件 (受理 0, 却下 6, 受理率 0.000)
- 履歴: req 36 件, 平均受理率 0.500
- 予測結果一覧 (model 列で IRL / RF を区別):
  |model|train|eval|split|prob|pred|true|hist_req|hist_rate|eval_req|eval_acc|eval_rej|
  |---|---|---|---|---|---|---|---|---|---|---|---|
  |IRL|6-9m|6-9m|-|0.480|0|0|18|0.5|5|0|5|
  |IRL|6-9m|9-12m|-|0.480|0|0|18|0.5|1|0|1|

### pierre-samuel.le-stang@corp.ovh.com

- グループ: Light-Low
- eval 合計: 6 件 (受理 0, 却下 6, 受理率 0.000)
- 履歴: req 84 件, 平均受理率 0.333
- 予測結果一覧 (model 列で IRL / RF を区別):
  |model|train|eval|split|prob|pred|true|hist_req|hist_rate|eval_req|eval_acc|eval_rej|
  |---|---|---|---|---|---|---|---|---|---|---|---|
  |IRL|0-3m|0-3m|-|0.460|0|0|12|0.3333333333333333|0|0|0|
  |IRL|0-3m|3-6m|-|0.460|0|0|12|0.3333333333333333|0|0|0|
  |IRL|0-3m|6-9m|-|0.460|0|0|12|0.3333333333333333|0|0|0|
  |IRL|0-3m|9-12m|-|0.460|0|0|12|0.3333333333333333|2|0|2|
  |IRL|6-9m|6-9m|-|0.479|0|0|12|0.3333333333333333|0|0|0|
  |IRL|6-9m|9-12m|-|0.479|0|0|12|0.3333333333333333|2|0|2|
  |IRL|9-12m|9-12m|-|0.480|0|0|12|0.3333333333333333|2|0|2|

### markgoddard86@gmail.com

- グループ: Light-Low
- eval 合計: 5 件 (受理 0, 却下 5, 受理率 0.000)
- 履歴: req 19 件, 平均受理率 0.053
- 予測結果一覧 (model 列で IRL / RF を区別):
  |model|train|eval|split|prob|pred|true|hist_req|hist_rate|eval_req|eval_acc|eval_rej|
  |---|---|---|---|---|---|---|---|---|---|---|---|
  |IRL|9-12m|9-12m|-|0.477|0|0|19|0.0526315789473684|5|0|5|

### kajinamit@oss.nttdata.com

- グループ: Light-Low
- eval 合計: 3 件 (受理 0, 却下 3, 受理率 0.000)
- 履歴: req 63 件, 平均受理率 0.048
- 予測結果一覧 (model 列で IRL / RF を区別):
  |model|train|eval|split|prob|pred|true|hist_req|hist_rate|eval_req|eval_acc|eval_rej|
  |---|---|---|---|---|---|---|---|---|---|---|---|
  |IRL|6-9m|6-9m|-|0.480|0|0|21|0.0476190476190476|1|0|1|
  |IRL|6-9m|9-12m|-|0.480|0|0|21|0.0476190476190476|1|0|1|
  |IRL|9-12m|9-12m|-|0.479|0|0|21|0.0476190476190476|1|0|1|

### openstack-ci@storpool.com

- グループ: Light-Low
- eval 合計: 3 件 (受理 0, 却下 3, 受理率 0.000)
- 履歴: req 49 件, 平均受理率 0.000
- 予測結果一覧 (model 列で IRL / RF を区別):
  |model|train|eval|split|prob|pred|true|hist_req|hist_rate|eval_req|eval_acc|eval_rej|
  |---|---|---|---|---|---|---|---|---|---|---|---|
  |IRL|9-12m|9-12m|-|0.478|0|0|49|0.0|3|0|3|

### kinpaa@gmail.com

- グループ: Light-Low
- eval 合計: 2 件 (受理 0, 却下 2, 受理率 0.000)
- 履歴: req 21 件, 平均受理率 0.000
- 予測結果一覧 (model 列で IRL / RF を区別):
  |model|train|eval|split|prob|pred|true|hist_req|hist_rate|eval_req|eval_acc|eval_rej|
  |---|---|---|---|---|---|---|---|---|---|---|---|
  |IRL|0-3m|0-3m|-|0.468|0|0|7|0.0|0|0|0|
  |IRL|0-3m|3-6m|-|0.468|0|0|7|0.0|1|0|1|
  |IRL|3-6m|3-6m|-|0.475|0|0|7|0.0|1|0|1|

### rene.ribaud@gmail.com

- グループ: Light-Low
- eval 合計: 2 件 (受理 0, 却下 2, 受理率 0.000)
- 履歴: req 6 件, 平均受理率 0.167
- 予測結果一覧 (model 列で IRL / RF を区別):
  |model|train|eval|split|prob|pred|true|hist_req|hist_rate|eval_req|eval_acc|eval_rej|
  |---|---|---|---|---|---|---|---|---|---|---|---|
  |IRL|9-12m|9-12m|-|0.474|0|0|6|0.1666666666666666|2|0|2|

### maksim.malchuk@gmail.com

- グループ: Light-High
- eval 合計: 1 件 (受理 1, 却下 0, 受理率 1.000)
- 履歴: req 6 件, 平均受理率 0.667
- 予測結果一覧 (model 列で IRL / RF を区別):
  |model|train|eval|split|prob|pred|true|hist_req|hist_rate|eval_req|eval_acc|eval_rej|
  |---|---|---|---|---|---|---|---|---|---|---|---|
  |IRL|6-9m|6-9m|-|0.479|0|0|3|0.6666666666666666|0|0|0|
  |IRL|6-9m|9-12m|-|0.479|0|1|3|0.6666666666666666|1|1|0|

### kchamart@redhat.com

- グループ: なし
- eval 合計: 0 件 (受理 0, 却下 0, 受理率 0.000)
- 履歴: req 0 件, 平均受理率 0.000
- 予測結果一覧 (model 列で IRL / RF を区別):
  |model|train|eval|split|prob|pred|true|hist_req|hist_rate|eval_req|eval_acc|eval_rej|
  |---|---|---|---|---|---|---|---|---|---|---|---|
  |RF|-|-|0-3m_to_6-9m|0.609|1|1|-|-|-|-|-|

### tobias.urdin@binero.com

- グループ: なし
- eval 合計: 0 件 (受理 0, 却下 0, 受理率 0.000)
- 履歴: req 0 件, 平均受理率 0.000
- 予測結果一覧 (model 列で IRL / RF を区別):
  |model|train|eval|split|prob|pred|true|hist_req|hist_rate|eval_req|eval_acc|eval_rej|
  |---|---|---|---|---|---|---|---|---|---|---|---|
  |RF|-|-|0-3m_to_6-9m|0.563|1|0|-|-|-|-|-|

## RF だけが的中した（IRL では当てられなかった／未カバー）開発者

- kchamart@redhat.com: RF 1/1 正解。IRL データなし。特徴追加で IRL でも対応したい。
- tobias.urdin@binero.com: RF 0/1 正解。IRL データなし。特徴追加で IRL でも対応したい。

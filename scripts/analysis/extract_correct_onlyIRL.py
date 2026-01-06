from pathlib import Path

import pandas as pd

base_dir= Path('./outputs/singleproject/irl_rf_10pattern_analysis')
print(base_dir)
# 1. CSVの読み込み
df_a = pd.read_csv(base_dir / 'irl_only_correct.csv')
df_b = pd.read_csv(base_dir / 'rf_only_correct.csv')

# 2. フィルタリング実行
# 意味: df_aの中で、dev_idが「df_bのdev_idリストに含まれていない(~)」行を取り出す
diff_df = df_a[~df_a['reviewer_id'].isin(df_b['reviewer_id'])]

# 3. 結果の確認と保存
print(diff_df)
diff_df.to_csv(base_dir / 'result.csv', index=False)
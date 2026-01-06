"""
IRLのみ正解に登場するが、RFのみ正解には登場しない開発者を抽出
"""

from pathlib import Path
import pandas as pd

# パス設定
BASE_DIR = Path("/Users/kazuki-h/research/multiproject_research")
IRL_ONLY_CSV = BASE_DIR / "outputs/irl_rf_10pattern_analysis/irl_only_correct.csv"
RF_ONLY_CSV = BASE_DIR / "outputs/irl_rf_10pattern_analysis/rf_only_correct.csv"
OUTPUT_CSV = BASE_DIR / "outputs/singleproject/irl_only_correct_analysis/irl_only_unique_developers.csv"
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# データ読み込み
irl_only_df = pd.read_csv(IRL_ONLY_CSV)
rf_only_df = pd.read_csv(RF_ONLY_CSV)

# 開発者IDのセット
irl_developers = set(irl_only_df['reviewer_id'].unique())
rf_developers = set(rf_only_df['reviewer_id'].unique())

# IRLのみ正解に"だけ"存在する開発者
irl_unique_developers = irl_developers - rf_developers

print("="*80)
print("IRL のみ正解に登場する開発者の分析")
print("="*80)
print(f"\nIRLのみ正解に登場する開発者数: {len(irl_developers)}名")
print(f"RFのみ正解に登場する開発者数: {len(rf_developers)}名")
print(f"両方に登場する開発者数: {len(irl_developers & rf_developers)}名")
print(f"IRLのみ正解に「だけ」存在する開発者: {len(irl_unique_developers)}名")

# IRLのみ正解にだけ存在する開発者のデータを抽出
irl_unique_df = irl_only_df[irl_only_df['reviewer_id'].isin(irl_unique_developers)]

# 開発者ごとに集約
unique_dev_summary = irl_unique_df.groupby('reviewer_id').agg({
    'true_label': 'first',
    'history_count': 'first',
    'history_rate': 'first',
    'pattern': lambda x: ', '.join(x.tolist()),  # 登場パターンをリスト化
    'irl_prob': 'mean',
    'rf_prob': 'mean'
}).rename(columns={'pattern': 'patterns'})

# 登場回数を追加
unique_dev_summary['appearance_count'] = irl_unique_df.groupby('reviewer_id').size()

# 継続/離脱のラベル
unique_dev_summary['label'] = unique_dev_summary['true_label'].map({0: '離脱', 1: '継続'})

# ソート（登場回数の多い順）
unique_dev_summary = unique_dev_summary.sort_values('appearance_count', ascending=False)

print("\n" + "="*80)
print("IRLのみ正解に「だけ」存在する開発者の詳細")
print("="*80)

# 継続/離脱の内訳
continue_count = (unique_dev_summary['true_label'] == 1).sum()
leave_count = (unique_dev_summary['true_label'] == 0).sum()

print(f"\n継続者: {continue_count}名 ({continue_count/len(unique_dev_summary)*100:.1f}%)")
print(f"離脱者: {leave_count}名 ({leave_count/len(unique_dev_summary)*100:.1f}%)")

print(f"\n履歴依頼数:")
print(f"  平均={unique_dev_summary['history_count'].mean():.1f}, 中央値={unique_dev_summary['history_count'].median():.1f}")
print(f"  最小={unique_dev_summary['history_count'].min():.0f}, 最大={unique_dev_summary['history_count'].max():.0f}")

print(f"\n履歴承諾率:")
print(f"  平均={unique_dev_summary['history_rate'].mean():.3f}, 中央値={unique_dev_summary['history_rate'].median():.3f}")

print(f"\nIRL予測確率:")
print(f"  平均={unique_dev_summary['irl_prob'].mean():.3f}")

print(f"\nRF予測確率:")
print(f"  平均={unique_dev_summary['rf_prob'].mean():.3f}")

# 全開発者リスト
print("\n" + "="*80)
print("開発者一覧（登場回数順）")
print("="*80)

for idx, (dev_id, row) in enumerate(unique_dev_summary.iterrows(), 1):
    print(f"\n{idx}. {dev_id}")
    print(f"   {row['label']}, 登場回数={row['appearance_count']:.0f}, "
          f"履歴={row['history_count']:.0f}件, 承諾率={row['history_rate']:.3f}")
    print(f"   IRL確率={row['irl_prob']:.3f}, RF確率={row['rf_prob']:.3f}")
    print(f"   パターン: {row['patterns']}")

# CSVに保存
unique_dev_summary.to_csv(OUTPUT_CSV)
print(f"\n保存先: {OUTPUT_CSV}")

print("\n" + "="*80)
print("完了！")
print("="*80)

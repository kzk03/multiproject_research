"""
IRL と RF の特徴量重要度データを1つのCSVにまとめる
"""

from pathlib import Path
import json
import pandas as pd

# パス設定
BASE_DIR = Path("/Users/kazuki-h/research/multiproject_research")
IRL_BASE = BASE_DIR / "results/review_continuation_cross_eval_nova"
RF_CSV = BASE_DIR / "outputs/analysis_data/feature_importance_comparison/rf_importance_by_period.csv"
OUTPUT_CSV = BASE_DIR / "outputs/analysis_data/feature_importance_comparison/irl_rf_merged_importance.csv"

TRAIN_PERIODS = ['0-3m', '3-6m', '6-9m', '9-12m']

def load_irl_data():
    """IRLの訓練期間別データを読み込み"""
    all_data = []

    for period in TRAIN_PERIODS:
        json_path = IRL_BASE / f"train_{period}" / "feature_importance" / "gradient_importance.json"

        with open(json_path, 'r') as f:
            data = json.load(f)

        for key, value in data.get('state_importance', {}).items():
            all_data.append({
                'feature': key,
                'irl_importance': value,
                'irl_importance_abs': abs(value),
                'period': period,
                'category': 'state'
            })

        for key, value in data.get('action_importance', {}).items():
            all_data.append({
                'feature': key,
                'irl_importance': value,
                'irl_importance_abs': abs(value),
                'period': period,
                'category': 'action'
            })

    return pd.DataFrame(all_data)


def load_rf_data():
    """RFの訓練期間別データを読み込み"""
    rf_df = pd.read_csv(RF_CSV)

    # State vs Action分類
    state_features = ['経験日数', '総コミット数', '総レビュー数', '最近の活動頻度',
                     '平均活動間隔', 'レビュー負荷', '最近の受諾率', '活動トレンド',
                     '協力スコア', 'コード品質スコア']

    rf_df['category'] = rf_df['feature'].apply(
        lambda x: 'state' if x in state_features else 'action'
    )
    rf_df = rf_df.rename(columns={'importance': 'rf_importance'})

    return rf_df


def merge_data(irl_df, rf_df):
    """IRLとRFのデータをマージ"""
    # マージキー: feature + period
    merged = pd.merge(
        irl_df,
        rf_df[['feature', 'period', 'rf_importance']],
        on=['feature', 'period'],
        how='outer'
    )

    # カラム順序を整理
    merged = merged[['period', 'category', 'feature', 'irl_importance', 'irl_importance_abs', 'rf_importance']]

    # 期間と特徴量でソート
    period_order = {'0-3m': 0, '3-6m': 1, '6-9m': 2, '9-12m': 3}
    merged['period_order'] = merged['period'].map(period_order)
    merged = merged.sort_values(['period_order', 'category', 'feature'])
    merged = merged.drop('period_order', axis=1)

    return merged


def main():
    print("="*80)
    print("IRL と RF の特徴量重要度データを統合")
    print("="*80)

    # データ読み込み
    irl_df = load_irl_data()
    rf_df = load_rf_data()

    print(f"\nIRLデータ数: {len(irl_df)}")
    print(f"RFデータ数: {len(rf_df)}")

    # マージ
    merged_df = merge_data(irl_df, rf_df)

    print(f"\nマージ後データ数: {len(merged_df)}")
    print(f"\nカラム: {list(merged_df.columns)}")
    print(f"\n最初の10行:")
    print(merged_df.head(10))

    # CSV保存
    merged_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n保存完了: {OUTPUT_CSV}")

    # 統計情報
    print("\n" + "="*80)
    print("統計情報")
    print("="*80)

    for period in TRAIN_PERIODS:
        period_data = merged_df[merged_df['period'] == period]
        print(f"\n【{period}】")
        print(f"  データ数: {len(period_data)}")
        print(f"  State特徴量: {len(period_data[period_data['category'] == 'state'])}")
        print(f"  Action特徴量: {len(period_data[period_data['category'] == 'action'])}")

    print("\n" + "="*80)
    print("完了！")
    print("="*80)


if __name__ == "__main__":
    main()

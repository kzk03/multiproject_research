"""
IRL vs RF の特徴量重要度を折れ線グラフで比較（状態・行動統合版）

状態特徴量と行動特徴量を分けずに、全特徴量を1つのグラフに表示
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meirio']
plt.rcParams['font.family'] = 'sans-serif'

# パス設定
BASE_DIR = Path("/Users/kazuki-h/research/multiproject_research")
IRL_BASE = BASE_DIR / "results/review_continuation_cross_eval_nova"
RF_CSV = BASE_DIR / "outputs/analysis_data/feature_importance_comparison/rf_importance_by_period.csv"
OUTPUT_DIR = BASE_DIR / "outputs/analysis_data/feature_importance_comparison"

TRAIN_PERIODS = ['0-3m', '3-6m', '6-9m', '9-12m']

# 特徴量ごとに固定色を割り当て
FEATURE_COLORS = {
    # State features
    '経験日数': '#1f77b4',
    '総コミット数': '#ff7f0e',
    '総レビュー数': '#2ca02c',
    '最近の活動頻度': '#d62728',
    '平均活動間隔': '#9467bd',
    'レビュー負荷': '#8c564b',
    '最近の受諾率': '#e377c2',
    '活動トレンド': '#7f7f7f',
    '協力スコア': '#bcbd22',
    'コード品質スコア': '#17becf',
    # Action features
    '応答速度': '#aec7e8',
    '協力度': '#ffbb78',
    '強度（ファイル数）': '#98df8a',
    'レビュー規模（行数）': '#ff9896',
}

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
                'value': value,  # 符号付き
                'period': period,
                'category': 'state'
            })

        for key, value in data.get('action_importance', {}).items():
            all_data.append({
                'feature': key,
                'value': value,
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
    rf_df = rf_df.rename(columns={'importance': 'value'})

    return rf_df


def create_merged_line_chart(irl_df, rf_df):
    """状態・行動統合版の折れ線グラフ（IRL vs RF 横並び）"""

    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    fig.suptitle('IRL vs RF Case2: 全特徴量の重要度推移（状態・行動統合版）', fontsize=20, y=0.995)

    # (A) IRL 全特徴量
    ax = axes[0]

    for feature in sorted(irl_df['feature'].unique()):
        feature_data = irl_df[irl_df['feature'] == feature]
        feature_data = feature_data.sort_values('period')
        color = FEATURE_COLORS.get(feature, '#000000')
        ax.plot(range(len(TRAIN_PERIODS)), feature_data['value'],
               marker='o', label=feature, color=color, linewidth=2.5, markersize=8)

    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xticks(range(len(TRAIN_PERIODS)))
    ax.set_xticklabels(TRAIN_PERIODS, fontsize=13)
    ax.set_ylabel('勾配値（符号付き）', fontsize=14)
    ax.set_xlabel('訓練期間', fontsize=14)
    ax.set_title('(A) IRL: 全特徴量の推移', fontsize=16, pad=10, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, ncol=2, framealpha=0.9)
    ax.grid(alpha=0.3)

    # (B) RF Case2 全特徴量
    ax = axes[1]

    for feature in sorted(rf_df['feature'].unique()):
        feature_data = rf_df[rf_df['feature'] == feature]
        feature_data = feature_data.sort_values('period')
        color = FEATURE_COLORS.get(feature, '#000000')
        ax.plot(range(len(TRAIN_PERIODS)), feature_data['value'],
               marker='s', label=feature, color=color, linewidth=2.5, markersize=8)

    ax.set_xticks(range(len(TRAIN_PERIODS)))
    ax.set_xticklabels(TRAIN_PERIODS, fontsize=13)
    ax.set_ylabel('Gini重要度', fontsize=14)
    ax.set_xlabel('訓練期間', fontsize=14)
    ax.set_title('(B) RF Case2: 全特徴量の推移', fontsize=16, pad=10, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, ncol=2, framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    output_path = OUTPUT_DIR / "irl_vs_rf_line_chart_merged.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n折れ線グラフ（統合版）を保存: {output_path}")
    plt.close()


def create_top_features_comparison(irl_df, rf_df, top_n=8):
    """重要度上位N個の特徴量のみを抽出して比較"""

    # IRLの平均重要度（絶対値）でTop N
    irl_avg = irl_df.groupby('feature')['value'].apply(lambda x: x.abs().mean()).sort_values(ascending=False)
    top_features = irl_avg.head(top_n).index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    fig.suptitle(f'IRL vs RF Case2: 重要度上位{top_n}特徴量の推移', fontsize=20, y=0.995)

    # (A) IRL Top N
    ax = axes[0]

    for feature in top_features:
        feature_data = irl_df[irl_df['feature'] == feature]
        feature_data = feature_data.sort_values('period')
        color = FEATURE_COLORS.get(feature, '#000000')
        ax.plot(range(len(TRAIN_PERIODS)), feature_data['value'].abs(),
               marker='o', label=feature, color=color, linewidth=3, markersize=10)

    ax.set_xticks(range(len(TRAIN_PERIODS)))
    ax.set_xticklabels(TRAIN_PERIODS, fontsize=13)
    ax.set_ylabel('勾配値（絶対値）', fontsize=14)
    ax.set_xlabel('訓練期間', fontsize=14)
    ax.set_title(f'(A) IRL: 重要度上位{top_n}特徴量', fontsize=16, pad=10, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(alpha=0.3)

    # (B) RF Top N（同じ特徴量）
    ax = axes[1]

    for feature in top_features:
        feature_data = rf_df[rf_df['feature'] == feature]
        if len(feature_data) > 0:
            feature_data = feature_data.sort_values('period')
            color = FEATURE_COLORS.get(feature, '#000000')
            ax.plot(range(len(TRAIN_PERIODS)), feature_data['value'],
                   marker='s', label=feature, color=color, linewidth=3, markersize=10)

    ax.set_xticks(range(len(TRAIN_PERIODS)))
    ax.set_xticklabels(TRAIN_PERIODS, fontsize=13)
    ax.set_ylabel('Gini重要度', fontsize=14)
    ax.set_xlabel('訓練期間', fontsize=14)
    ax.set_title(f'(B) RF Case2: 重要度上位{top_n}特徴量（IRLベース）', fontsize=16, pad=10, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    output_path = OUTPUT_DIR / f"irl_vs_rf_top{top_n}_merged.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"上位{top_n}特徴量比較を保存: {output_path}")
    plt.close()

    print(f"\nIRL重要度上位{top_n}特徴量:")
    for i, (feature, value) in enumerate(irl_avg.head(top_n).items(), 1):
        print(f"  {i}. {feature}: {value:.6f}")


def create_overlay_comparison(irl_df, rf_df):
    """IRL と RF を同じグラフに重ねて表示（主要特徴量のみ）"""

    # 両方で重要な特徴量を選択
    selected_features = [
        '総レビュー数', '総コミット数', '最近の活動頻度', '平均活動間隔',
        '協力度', '応答速度', '最近の受諾率', 'レビュー負荷'
    ]

    fig, axes = plt.subplots(2, 4, figsize=(28, 12))
    fig.suptitle('IRL vs RF Case2: 主要特徴量の直接比較（重ね合わせ）', fontsize=20, y=0.995)

    for idx, feature in enumerate(selected_features):
        row = idx // 4
        col = idx % 4
        ax = axes[row, col]

        feature_color = FEATURE_COLORS.get(feature, '#000000')

        # IRLデータ（絶対値）
        irl_feature = irl_df[irl_df['feature'] == feature]
        if len(irl_feature) > 0:
            irl_feature = irl_feature.sort_values('period')
            ax.plot(range(len(TRAIN_PERIODS)), irl_feature['value'].abs(),
                   marker='o', label='IRL（勾配絶対値）', color=feature_color,
                   linewidth=3.5, markersize=12, alpha=0.8, linestyle='-')

        # RFデータ（正規化）
        rf_feature = rf_df[rf_df['feature'] == feature]
        if len(rf_feature) > 0:
            rf_feature = rf_feature.sort_values('period')
            # 同じスケールに正規化
            rf_max = rf_df['value'].max()
            irl_max = irl_df['value'].abs().max()
            rf_normalized = rf_feature['value'] * (irl_max / rf_max)

            ax.plot(range(len(TRAIN_PERIODS)), rf_normalized,
                   marker='s', label='RF（Gini・正規化）', color=feature_color,
                   linewidth=3.5, markersize=12, alpha=0.5, linestyle='--')

        ax.set_xticks(range(len(TRAIN_PERIODS)))
        ax.set_xticklabels(TRAIN_PERIODS, fontsize=11)
        ax.set_ylabel('正規化重要度', fontsize=12)
        ax.set_xlabel('訓練期間', fontsize=12)
        ax.set_title(feature, fontsize=14, pad=8, fontweight='bold', color=feature_color)
        ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    output_path = OUTPUT_DIR / "irl_vs_rf_overlay_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"重ね合わせ比較を保存: {output_path}")
    plt.close()


def main():
    print("="*80)
    print("IRL vs RF Case2: 折れ線グラフ比較（状態・行動統合版）")
    print("="*80)

    # データ読み込み
    irl_df = load_irl_data()
    rf_df = load_rf_data()

    print(f"\nIRLデータ数: {len(irl_df)}")
    print(f"RFデータ数: {len(rf_df)}")

    # 可視化
    create_merged_line_chart(irl_df, rf_df)
    create_top_features_comparison(irl_df, rf_df, top_n=8)
    create_overlay_comparison(irl_df, rf_df)

    print("\n" + "="*80)
    print("完了！")
    print("="*80)


if __name__ == "__main__":
    main()

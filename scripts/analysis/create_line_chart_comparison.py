"""
IRL vs RF の特徴量重要度を折れ線グラフで比較

同じ特徴量には同じ色を割り当てて比較しやすくする
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
    '総レビュー依頼数': '#ff7f0e',
    '総レビュー数': '#2ca02c',
    '最近の活動頻度': '#d62728',
    '平均活動間隔': '#9467bd',
    'レビュー負荷': '#8c564b',
    '最近の受諾率': '#e377c2',
    '活動トレンド': '#7f7f7f',
    '協力スコア': '#bcbd22',
    '総承諾率': '#17becf',
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
    state_features = ['経験日数', '総レビュー依頼数', '総レビュー数', '最近の活動頻度',
                     '平均活動間隔', 'レビュー負荷', '最近の受諾率', '活動トレンド',
                     '協力スコア', '総承諾率']
    
    rf_df['category'] = rf_df['feature'].apply(
        lambda x: 'state' if x in state_features else 'action'
    )
    rf_df = rf_df.rename(columns={'importance': 'value'})
    
    return rf_df


def create_line_chart_comparison(irl_df, rf_df):
    """折れ線グラフでIRLとRFを比較（色統一版）"""
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle('IRL vs RF Case2: 特徴量重要度の時系列推移（色統一版）', fontsize=20, y=0.995)
    
    # (A) IRL 状態特徴量
    ax = axes[0, 0]
    irl_state = irl_df[irl_df['category'] == 'state']
    
    for feature in sorted(irl_state['feature'].unique()):
        feature_data = irl_state[irl_state['feature'] == feature]
        feature_data = feature_data.sort_values('period')
        color = FEATURE_COLORS.get(feature, '#000000')
        ax.plot(range(len(TRAIN_PERIODS)), feature_data['value'], 
               marker='o', label=feature, color=color, linewidth=2.5, markersize=8)
    
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xticks(range(len(TRAIN_PERIODS)))
    ax.set_xticklabels(TRAIN_PERIODS, fontsize=12)
    ax.set_ylabel('勾配値（符号付き）', fontsize=13)
    ax.set_xlabel('訓練期間', fontsize=13)
    ax.set_title('(A) IRL: 状態特徴量の推移', fontsize=15, pad=10, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, ncol=2, framealpha=0.9)
    ax.grid(alpha=0.3)
    
    # (B) IRL 行動特徴量
    ax = axes[0, 1]
    irl_action = irl_df[irl_df['category'] == 'action']
    
    for feature in sorted(irl_action['feature'].unique()):
        feature_data = irl_action[irl_action['feature'] == feature]
        feature_data = feature_data.sort_values('period')
        color = FEATURE_COLORS.get(feature, '#000000')
        ax.plot(range(len(TRAIN_PERIODS)), feature_data['value'], 
               marker='o', label=feature, color=color, linewidth=2.5, markersize=8)
    
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xticks(range(len(TRAIN_PERIODS)))
    ax.set_xticklabels(TRAIN_PERIODS, fontsize=12)
    ax.set_ylabel('勾配値（符号付き）', fontsize=13)
    ax.set_xlabel('訓練期間', fontsize=13)
    ax.set_title('(B) IRL: 行動特徴量の推移', fontsize=15, pad=10, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3)
    
    # (C) RF Case2 状態特徴量
    ax = axes[1, 0]
    rf_state = rf_df[rf_df['category'] == 'state']
    
    for feature in sorted(rf_state['feature'].unique()):
        feature_data = rf_state[rf_state['feature'] == feature]
        feature_data = feature_data.sort_values('period')
        color = FEATURE_COLORS.get(feature, '#000000')
        ax.plot(range(len(TRAIN_PERIODS)), feature_data['value'], 
               marker='s', label=feature, color=color, linewidth=2.5, markersize=8)
    
    ax.set_xticks(range(len(TRAIN_PERIODS)))
    ax.set_xticklabels(TRAIN_PERIODS, fontsize=12)
    ax.set_ylabel('Gini重要度', fontsize=13)
    ax.set_xlabel('訓練期間', fontsize=13)
    ax.set_title('(C) RF Case2: 状態特徴量の推移', fontsize=15, pad=10, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, ncol=2, framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # (D) RF Case2 行動特徴量
    ax = axes[1, 1]
    rf_action = rf_df[rf_df['category'] == 'action']
    
    for feature in sorted(rf_action['feature'].unique()):
        feature_data = rf_action[rf_action['feature'] == feature]
        feature_data = feature_data.sort_values('period')
        color = FEATURE_COLORS.get(feature, '#000000')
        ax.plot(range(len(TRAIN_PERIODS)), feature_data['value'], 
               marker='s', label=feature, color=color, linewidth=2.5, markersize=8)
    
    ax.set_xticks(range(len(TRAIN_PERIODS)))
    ax.set_xticklabels(TRAIN_PERIODS, fontsize=12)
    ax.set_ylabel('Gini重要度', fontsize=13)
    ax.set_xlabel('訓練期間', fontsize=13)
    ax.set_title('(D) RF Case2: 行動特徴量の推移', fontsize=15, pad=10, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    output_path = OUTPUT_DIR / "irl_vs_rf_line_chart_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n折れ線グラフを保存: {output_path}")
    plt.close()


def create_selected_features_comparison(irl_df, rf_df):
    """主要特徴量のみを抜き出して比較（色統一版）"""
    
    # 重要な特徴量を選択
    selected_features = [
        '総レビュー数', '総レビュー依頼数', '最近の活動頻度', '平均活動間隔',
        '協力度', '応答速度', '最近の受諾率', 'レビュー規模（行数）'
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    fig.suptitle('IRL vs RF Case2: 主要特徴量の時系列比較（同色で対応）', fontsize=18, y=0.995)
    
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
                   linewidth=3, markersize=12, alpha=0.8, linestyle='-')
        
        # RFデータ（正規化して同スケールに）
        rf_feature = rf_df[rf_df['feature'] == feature]
        if len(rf_feature) > 0:
            rf_feature = rf_feature.sort_values('period')
            # RFの値を同じスケールに正規化
            rf_max = rf_df['value'].max()
            irl_max = irl_df['value'].abs().max()
            rf_normalized = rf_feature['value'] * (irl_max / rf_max)
            
            ax.plot(range(len(TRAIN_PERIODS)), rf_normalized, 
                   marker='s', label='RF（Gini・正規化）', color=feature_color, 
                   linewidth=3, markersize=12, alpha=0.5, linestyle='--')
        
        ax.set_xticks(range(len(TRAIN_PERIODS)))
        ax.set_xticklabels(TRAIN_PERIODS, fontsize=10)
        ax.set_ylabel('正規化重要度', fontsize=11)
        ax.set_xlabel('訓練期間', fontsize=11)
        ax.set_title(feature, fontsize=13, pad=8, fontweight='bold', color=feature_color)
        ax.legend(loc='best', fontsize=9)
        ax.grid(alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    output_path = OUTPUT_DIR / "irl_vs_rf_selected_features_line.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"主要特徴量比較を保存: {output_path}")
    plt.close()


def analyze_trends(irl_df, rf_df):
    """トレンド分析"""
    
    print("\n" + "="*80)
    print("時系列トレンド分析（傾き）")
    print("="*80)
    
    def calc_trend(df, feature):
        """線形トレンドを計算（傾き）"""
        feature_data = df[df['feature'] == feature].sort_values('period')
        if len(feature_data) < 2:
            return 0
        x = np.arange(len(feature_data))
        y = feature_data['value'].values
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    print("\n【IRL: 増加トレンドの特徴量（Top 5）】")
    irl_trends = []
    for feature in irl_df['feature'].unique():
        trend = calc_trend(irl_df, feature)
        irl_trends.append((feature, trend))
    
    irl_trends.sort(key=lambda x: x[1], reverse=True)
    for feature, trend in irl_trends[:5]:
        print(f"  {feature:25s}: 傾き={trend:+.6f}")
    
    print("\n【IRL: 減少トレンドの特徴量（Top 5）】")
    for feature, trend in sorted(irl_trends, key=lambda x: x[1])[:5]:
        print(f"  {feature:25s}: 傾き={trend:+.6f}")
    
    print("\n【RF: 増加トレンドの特徴量（Top 5）】")
    rf_trends = []
    for feature in rf_df['feature'].unique():
        trend = calc_trend(rf_df, feature)
        rf_trends.append((feature, trend))
    
    rf_trends.sort(key=lambda x: x[1], reverse=True)
    for feature, trend in rf_trends[:5]:
        print(f"  {feature:25s}: 傾き={trend:+.6f}")
    
    print("\n【RF: 減少トレンドの特徴量（Top 5）】")
    for feature, trend in sorted(rf_trends, key=lambda x: x[1])[:5]:
        print(f"  {feature:25s}: 傾き={trend:+.6f}")


def main():
    print("="*80)
    print("IRL vs RF Case2: 折れ線グラフ比較（色統一版）")
    print("="*80)
    
    # データ読み込み
    irl_df = load_irl_data()
    rf_df = load_rf_data()
    
    print(f"\nIRLデータ数: {len(irl_df)}")
    print(f"RFデータ数: {len(rf_df)}")
    
    # 可視化
    create_line_chart_comparison(irl_df, rf_df)
    create_selected_features_comparison(irl_df, rf_df)
    
    # トレンド分析
    analyze_trends(irl_df, rf_df)
    
    print("\n" + "="*80)
    print("完了！")
    print("="*80)


if __name__ == "__main__":
    main()

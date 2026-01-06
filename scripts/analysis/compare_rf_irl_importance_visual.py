"""
RFとIRLの特徴量重要度を並べて比較可視化
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
RF_CSV = BASE_DIR / "outputs/analysis_data/feature_importance_comparison/rf_case2_feature_importance.csv"
IRL_JSON = BASE_DIR / "results/review_continuation_cross_eval_nova/average_feature_importance/gradient_importance_average.json"
OUTPUT_DIR = BASE_DIR / "outputs/analysis_data/feature_importance_comparison"

def load_data():
    """データを読み込み"""
    # RF重要度
    rf_df = pd.read_csv(RF_CSV)
    
    # IRL勾配重要度
    with open(IRL_JSON, 'r') as f:
        irl_data = json.load(f)
    
    # IRLデータを整形（state_importanceとaction_importanceを結合）
    irl_features = []
    irl_values = []
    
    for key, value in irl_data.get('state_importance', {}).items():
        irl_features.append(key)
        irl_values.append(abs(value))  # 絶対値を取る
    
    for key, value in irl_data.get('action_importance', {}).items():
        irl_features.append(key)
        irl_values.append(abs(value))
    
    irl_df = pd.DataFrame({
        'feature': irl_features,
        'gradient': irl_values
    })
    
    return rf_df, irl_df


def create_comparison_plot(rf_df, irl_df):
    """RFとIRLの比較プロット"""
    
    # 共通の特徴量のみを抽出
    common_features = set(rf_df['feature']) & set(irl_df['feature'])
    
    print("="*80)
    print("RF vs IRL 特徴量重要度比較")
    print("="*80)
    print(f"\nRF特徴量数: {len(rf_df)}")
    print(f"IRL特徴量数: {len(irl_df)}")
    print(f"共通特徴量数: {len(common_features)}")
    
    # データを結合
    merged = pd.merge(
        rf_df[rf_df['feature'].isin(common_features)],
        irl_df[irl_df['feature'].isin(common_features)],
        on='feature',
        how='inner'
    )
    
    # 正規化（0-1スケール）
    merged['rf_norm'] = merged['importance'] / merged['importance'].max()
    merged['irl_norm'] = merged['gradient'] / merged['gradient'].max()
    
    # ソート（RF重要度順）
    merged = merged.sort_values('importance', ascending=True)
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    
    # 1. 横並び棒グラフ（正規化済み）
    ax = axes[0]
    
    y_pos = np.arange(len(merged))
    width = 0.35
    
    ax.barh(y_pos - width/2, merged['rf_norm'], width, 
            label='RF Case2 (Gini係数)', color='#3498db', alpha=0.8)
    ax.barh(y_pos + width/2, merged['irl_norm'], width, 
            label='IRL (勾配)', color='#e74c3c', alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(merged['feature'], fontsize=11)
    ax.set_xlabel('正規化重要度（最大値=1.0）', fontsize=12)
    ax.set_title('RF Case2 vs IRL: 特徴量重要度比較（正規化済み）', fontsize=14, pad=15)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    
    # 2. 散布図（相関）
    ax = axes[1]
    
    # カテゴリ分け（状態 vs 行動）
    state_features = ['経験日数', '総コミット数', '総レビュー数', '最近の活動頻度', 
                     '平均活動間隔', 'レビュー負荷', '最近の受諾率', '活動トレンド',
                     '協力スコア', 'コード品質スコア']
    
    merged['category'] = merged['feature'].apply(
        lambda x: 'State' if x in state_features else 'Action'
    )
    
    for category, color, marker in [('State', '#3498db', 'o'), ('Action', '#2ecc71', 's')]:
        subset = merged[merged['category'] == category]
        ax.scatter(subset['rf_norm'], subset['irl_norm'], 
                  c=color, s=150, alpha=0.7, marker=marker,
                  label=f'{category}特徴量', edgecolors='black', linewidth=1)
        
        # ラベル
        for _, row in subset.iterrows():
            ax.annotate(row['feature'], 
                       (row['rf_norm'], row['irl_norm']),
                       fontsize=9, alpha=0.8,
                       xytext=(5, 5), textcoords='offset points')
    
    # 対角線
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x')
    
    ax.set_xlabel('RF重要度（正規化）', fontsize=12)
    ax.set_ylabel('IRL重要度（正規化）', fontsize=12)
    ax.set_title('RF Case2 vs IRL: 重要度の相関', fontsize=14, pad=15)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / "rf_vs_irl_importance_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n可視化を保存: {output_path}")
    plt.close()
    
    # ランク変化の分析
    print("\n" + "="*80)
    print("ランク変化の分析")
    print("="*80)
    
    rf_ranks = merged.sort_values('importance', ascending=False).reset_index(drop=True)
    rf_ranks['rf_rank'] = range(1, len(rf_ranks) + 1)
    
    irl_ranks = merged.sort_values('gradient', ascending=False).reset_index(drop=True)
    irl_ranks['irl_rank'] = range(1, len(irl_ranks) + 1)
    
    rank_comparison = pd.merge(
        rf_ranks[['feature', 'rf_rank', 'importance']],
        irl_ranks[['feature', 'irl_rank', 'gradient']],
        on='feature'
    )
    rank_comparison['rank_change'] = rank_comparison['irl_rank'] - rank_comparison['rf_rank']
    rank_comparison = rank_comparison.sort_values('rank_change', key=abs, ascending=False)
    
    print("\nランク変化が大きい特徴量（上位10個）:")
    for idx, row in rank_comparison.head(10).iterrows():
        direction = "↑" if row['rank_change'] < 0 else "↓"
        print(f"{row['feature']:25s}: RF {row['rf_rank']:2d}位 → IRL {row['irl_rank']:2d}位 "
              f"({direction}{abs(row['rank_change']):2d}) | RF={row['importance']:.4f}, IRL={row['gradient']:.6f}")
    
    # CSV保存
    csv_path = OUTPUT_DIR / "rf_vs_irl_comparison_detailed.csv"
    rank_comparison.to_csv(csv_path, index=False)
    print(f"\n詳細CSVを保存: {csv_path}")
    
    return merged


def main():
    rf_df, irl_df = load_data()
    merged = create_comparison_plot(rf_df, irl_df)
    
    print("\n" + "="*80)
    print("完了！")
    print("="*80)


if __name__ == "__main__":
    main()

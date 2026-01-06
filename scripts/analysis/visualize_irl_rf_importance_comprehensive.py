"""
IRLとRFの特徴量重要度を包括的に比較可視化

両モデルの訓練期間別の重要度変化を並べて表示
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

def load_irl_importance():
    """IRLの各訓練期間の特徴量重要度を読み込み"""
    
    all_data = []
    
    for period in TRAIN_PERIODS:
        json_path = IRL_BASE / f"train_{period}" / "feature_importance" / "gradient_importance.json"
        
        if not json_path.exists():
            print(f"警告: {json_path} が見つかりません")
            continue
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # state_importanceとaction_importanceを結合
        for key, value in data.get('state_importance', {}).items():
            all_data.append({
                'feature': key,
                'gradient': abs(value),  # 絶対値
                'period': period,
                'model': 'IRL'
            })
        
        for key, value in data.get('action_importance', {}).items():
            all_data.append({
                'feature': key,
                'gradient': abs(value),
                'period': period,
                'model': 'IRL'
            })
    
    return pd.DataFrame(all_data)


def load_rf_importance():
    """RFの訓練期間別重要度を読み込み"""
    rf_df = pd.read_csv(RF_CSV)
    rf_df['model'] = 'RF'
    rf_df = rf_df.rename(columns={'importance': 'value'})
    return rf_df[['feature', 'value', 'period', 'model']]


def create_side_by_side_comparison(irl_df, rf_df):
    """IRLとRFを横並びで比較"""
    
    # IRLのデータを正規化用に準備
    irl_df = irl_df.rename(columns={'gradient': 'value'})
    
    # 両方を結合
    combined = pd.concat([irl_df, rf_df], ignore_index=True)
    
    # 各モデル・期間ごとに正規化
    normalized = []
    for (model, period), group in combined.groupby(['model', 'period']):
        group['value_norm'] = group['value'] / group['value'].max()
        normalized.append(group)
    
    combined_norm = pd.concat(normalized, ignore_index=True)
    
    # 可視化
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle('IRL vs RF Case2: 訓練期間別 特徴量重要度の比較', fontsize=20, y=0.995)
    
    for col_idx, period in enumerate(TRAIN_PERIODS):
        # IRL
        ax = axes[0, col_idx]
        irl_period = combined_norm[(combined_norm['model'] == 'IRL') & 
                                    (combined_norm['period'] == period)]
        irl_period = irl_period.sort_values('value_norm', ascending=True)
        
        state_features = ['経験日数', '総コミット数', '総レビュー数', '最近の活動頻度',
                         '平均活動間隔', 'レビュー負荷', '最近の受諾率', '活動トレンド',
                         '協力スコア', 'コード品質スコア']
        colors_irl = ['#3498db' if f in state_features else '#2ecc71' 
                      for f in irl_period['feature']]
        
        ax.barh(range(len(irl_period)), irl_period['value_norm'], color=colors_irl)
        ax.set_yticks(range(len(irl_period)))
        ax.set_yticklabels(irl_period['feature'], fontsize=9)
        ax.set_xlabel('正規化重要度', fontsize=10)
        ax.set_title(f'IRL: {period}', fontsize=13, pad=10)
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(0, 1.05)
        
        # RF
        ax = axes[1, col_idx]
        rf_period = combined_norm[(combined_norm['model'] == 'RF') & 
                                   (combined_norm['period'] == period)]
        rf_period = rf_period.sort_values('value_norm', ascending=True)
        
        colors_rf = ['#3498db' if f in state_features else '#2ecc71' 
                     for f in rf_period['feature']]
        
        ax.barh(range(len(rf_period)), rf_period['value_norm'], color=colors_rf)
        ax.set_yticks(range(len(rf_period)))
        ax.set_yticklabels(rf_period['feature'], fontsize=9)
        ax.set_xlabel('正規化重要度', fontsize=10)
        ax.set_title(f'RF Case2: {period}', fontsize=13, pad=10)
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(0, 1.05)
    
    # 凡例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='状態特徴量（State）'),
        Patch(facecolor='#2ecc71', label='行動特徴量（Action）')
    ]
    fig.legend(handles=legend_elements, loc='upper center', 
               bbox_to_anchor=(0.5, 0.98), ncol=2, fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    output_path = OUTPUT_DIR / "irl_vs_rf_comprehensive_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n包括的比較を保存: {output_path}")
    plt.close()


def create_heatmap_comparison(irl_df, rf_df):
    """IRLとRFのヒートマップ比較"""
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle('IRL vs RF Case2: 特徴量重要度ヒートマップ比較', fontsize=18, y=0.98)
    
    # IRL
    ax = axes[0]
    irl_pivot = irl_df.pivot(index='feature', columns='period', values='gradient')
    irl_pivot['mean'] = irl_pivot.mean(axis=1)
    irl_pivot = irl_pivot.sort_values('mean', ascending=False).drop('mean', axis=1)
    
    sns.heatmap(irl_pivot.head(10), annot=True, fmt='.4f', cmap='Reds',
                cbar_kws={'label': '勾配値（絶対値）'}, ax=ax,
                linewidths=0.5, linecolor='gray')
    ax.set_title('IRL: 上位10特徴量', fontsize=15, pad=15)
    ax.set_xlabel('訓練期間', fontsize=12)
    ax.set_ylabel('特徴量', fontsize=12)
    
    # RF
    ax = axes[1]
    rf_pivot = rf_df.pivot(index='feature', columns='period', values='value')
    rf_pivot['mean'] = rf_pivot.mean(axis=1)
    rf_pivot = rf_pivot.sort_values('mean', ascending=False).drop('mean', axis=1)
    
    sns.heatmap(rf_pivot.head(10), annot=True, fmt='.4f', cmap='Blues',
                cbar_kws={'label': 'Gini重要度'}, ax=ax,
                linewidths=0.5, linecolor='gray')
    ax.set_title('RF Case2: 上位10特徴量', fontsize=15, pad=15)
    ax.set_xlabel('訓練期間', fontsize=12)
    ax.set_ylabel('特徴量', fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = OUTPUT_DIR / "irl_vs_rf_heatmap_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"ヒートマップ比較を保存: {output_path}")
    plt.close()


def analyze_rank_changes(irl_df, rf_df):
    """各期間でのランク変化を分析"""
    
    print("\n" + "="*80)
    print("期間別ランク変化分析")
    print("="*80)
    
    for period in TRAIN_PERIODS:
        print(f"\n【{period}】")
        
        irl_period = irl_df[irl_df['period'] == period].copy()
        rf_period = rf_df[rf_df['period'] == period].copy()
        
        irl_period = irl_period.sort_values('gradient', ascending=False).reset_index(drop=True)
        irl_period['irl_rank'] = range(1, len(irl_period) + 1)
        
        rf_period = rf_period.sort_values('value', ascending=False).reset_index(drop=True)
        rf_period['rf_rank'] = range(1, len(rf_period) + 1)
        
        comparison = pd.merge(
            irl_period[['feature', 'irl_rank', 'gradient']],
            rf_period[['feature', 'rf_rank', 'value']],
            on='feature'
        )
        comparison['rank_diff'] = comparison['irl_rank'] - comparison['rf_rank']
        comparison = comparison.sort_values('rank_diff', key=abs, ascending=False)
        
        print("大きなランク変化（Top 5）:")
        for idx, row in comparison.head(5).iterrows():
            direction = "←IRL優位" if row['rank_diff'] < 0 else "←RF優位"
            print(f"  {row['feature']:25s}: IRL {row['irl_rank']:2d}位 vs RF {row['rf_rank']:2d}位 "
                  f"(差{row['rank_diff']:+3d}) {direction}")


def main():
    print("="*80)
    print("IRL vs RF Case2: 包括的特徴量重要度比較")
    print("="*80)
    
    # データ読み込み
    irl_df = load_irl_importance()
    rf_df = load_rf_importance()
    
    print(f"\nIRLデータ数: {len(irl_df)}")
    print(f"RFデータ数: {len(rf_df)}")
    
    # 可視化
    create_side_by_side_comparison(irl_df, rf_df)
    create_heatmap_comparison(irl_df, rf_df)
    
    # 分析
    analyze_rank_changes(irl_df, rf_df)
    
    print("\n" + "="*80)
    print("完了！")
    print("="*80)


if __name__ == "__main__":
    main()

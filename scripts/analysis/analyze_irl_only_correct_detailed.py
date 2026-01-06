"""
IRLのみ正解の開発者の詳細分析

IRLがRFよりも優れている予測ケースの特性を深掘りする。
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meirio']
plt.rcParams['font.family'] = 'sans-serif'

# パス設定
BASE_DIR = Path("/Users/kazuki-h/research/multiproject_research")
IRL_ONLY_CSV = BASE_DIR / "outputs/irl_rf_10pattern_analysis/irl_only_correct.csv"
OUTPUT_DIR = BASE_DIR / "outputs/singleproject/irl_only_correct_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_and_analyze():
    """IRLのみ正解データを読み込んで分析"""
    df = pd.read_csv(IRL_ONLY_CSV)
    
    print("="*80)
    print("IRLのみ正解：詳細分析")
    print("="*80)
    print(f"\n総件数: {len(df)}")
    
    # 継続者/離脱者の内訳
    continue_df = df[df['true_label'] == 1]
    leave_df = df[df['true_label'] == 0]
    
    print(f"\n継続者を正しく予測: {len(continue_df)}件 ({len(continue_df)/len(df)*100:.1f}%)")
    print(f"離脱者を正しく予測: {len(leave_df)}件 ({len(leave_df)/len(df)*100:.1f}%)")
    
    # 履歴特性の分析
    print("\n" + "="*80)
    print("【履歴特性の分析】")
    print("="*80)
    
    print(f"\n履歴依頼数:")
    print(f"  全体: 平均={df['history_count'].mean():.1f}, 中央値={df['history_count'].median():.1f}")
    print(f"       最小={df['history_count'].min():.0f}, 最大={df['history_count'].max():.0f}")
    print(f"  継続者: 平均={continue_df['history_count'].mean():.1f}, 中央値={continue_df['history_count'].median():.1f}")
    print(f"  離脱者: 平均={leave_df['history_count'].mean():.1f}, 中央値={leave_df['history_count'].median():.1f}")
    
    print(f"\n履歴承諾率:")
    print(f"  全体: 平均={df['history_rate'].mean():.3f}, 中央値={df['history_rate'].median():.3f}")
    print(f"       最小={df['history_rate'].min():.3f}, 最大={df['history_rate'].max():.3f}")
    print(f"  継続者: 平均={continue_df['history_rate'].mean():.3f}, 中央値={continue_df['history_rate'].median():.3f}")
    print(f"  離脱者: 平均={leave_df['history_rate'].mean():.3f}, 中央値={leave_df['history_rate'].median():.3f}")
    
    # IRLとRFの確率の違い
    print("\n" + "="*80)
    print("【予測確率の分析】")
    print("="*80)
    
    df['prob_diff'] = df['irl_prob'] - df['rf_prob']
    
    print(f"\nIRL確率 - RF確率の差:")
    print(f"  平均={df['prob_diff'].mean():.3f}, 中央値={df['prob_diff'].median():.3f}")
    print(f"  最小={df['prob_diff'].min():.3f}, 最大={df['prob_diff'].max():.3f}")
    
    print(f"\n継続者（true_label=1）:")
    print(f"  IRL確率: 平均={continue_df['irl_prob'].mean():.3f}")
    print(f"  RF確率:  平均={continue_df['rf_prob'].mean():.3f}")
    print(f"  → IRLは継続者に対して慎重な確率（0.5付近）を出しているが、RFは低すぎる確率を出している")
    
    print(f"\n離脱者（true_label=0）:")
    print(f"  IRL確率: 平均={leave_df['irl_prob'].mean():.3f}")
    print(f"  RF確率:  平均={leave_df['rf_prob'].mean():.3f}")
    print(f"  → IRLは離脱者を正しく低確率で予測、RFは高確率を出してしまっている")
    
    # パターン別分析
    print("\n" + "="*80)
    print("【パターン別分析】")
    print("="*80)
    
    for pattern, group in df.groupby('pattern'):
        cont = (group['true_label'] == 1).sum()
        leave = (group['true_label'] == 0).sum()
        avg_hist = group['history_count'].mean()
        avg_rate = group['history_rate'].mean()
        print(f"\n{pattern}: {len(group)}件 (継続{cont}/離脱{leave})")
        print(f"  履歴依頼数: 平均={avg_hist:.1f}")
        print(f"  履歴承諾率: 平均={avg_rate:.3f}")
    
    # 特徴的な開発者の抽出
    print("\n" + "="*80)
    print("【特徴的なケースの分析】")
    print("="*80)
    
    # Case 1: 履歴が非常に少ない離脱者（RF は過去実績不足で誤判定）
    low_history_leave = leave_df[leave_df['history_count'] < 50]
    print(f"\nCase 1: 履歴が少ない離脱者（<50件）: {len(low_history_leave)}件")
    if len(low_history_leave) > 0:
        print(f"  平均履歴数: {low_history_leave['history_count'].mean():.1f}")
        print(f"  平均承諾率: {low_history_leave['history_rate'].mean():.3f}")
        print(f"  → RFは過去実績が少ないため判断を誤るが、IRLは活動パターンから離脱を予測")
    
    # Case 2: 承諾率が高い継続者（RF は静的指標で誤判定）
    high_rate_continue = continue_df[continue_df['history_rate'] > 0.2]
    print(f"\nCase 2: 承諾率が高い継続者（>0.2）: {len(high_rate_continue)}件")
    if len(high_rate_continue) > 0:
        print(f"  平均履歴数: {high_rate_continue['history_count'].mean():.1f}")
        print(f"  平均承諾率: {high_rate_continue['history_rate'].mean():.3f}")
        print(f"  → RFは静的な承諾率だけでは継続を予測できないが、IRLは協力的行動パターンを捉える")
    
    # Case 3: 中程度の履歴を持つ離脱者（最も難しいケース）
    mid_history_leave = leave_df[(leave_df['history_count'] >= 50) & (leave_df['history_count'] <= 200)]
    print(f"\nCase 3: 中程度の履歴を持つ離脱者（50-200件）: {len(mid_history_leave)}件")
    if len(mid_history_leave) > 0:
        print(f"  平均履歴数: {mid_history_leave['history_count'].mean():.1f}")
        print(f"  平均承諾率: {mid_history_leave['history_rate'].mean():.3f}")
        print(f"  → RFは実績があるため継続と誤判定、IRLは活動減少トレンドから離脱を予測")
    
    return df, continue_df, leave_df


def create_visualizations(df, continue_df, leave_df):
    """可視化を作成"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('IRLのみ正解: 特性分析', fontsize=16, y=0.995)
    
    # 1. 履歴依頼数の分布（継続 vs 離脱）
    ax = axes[0, 0]
    ax.hist([continue_df['history_count'], leave_df['history_count']], 
            bins=20, label=['継続者', '離脱者'], alpha=0.7, color=['blue', 'red'])
    ax.set_xlabel('履歴依頼数', fontsize=12)
    ax.set_ylabel('件数', fontsize=12)
    ax.set_title('履歴依頼数の分布', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. 履歴承諾率の分布
    ax = axes[0, 1]
    ax.hist([continue_df['history_rate'], leave_df['history_rate']], 
            bins=20, label=['継続者', '離脱者'], alpha=0.7, color=['blue', 'red'])
    ax.set_xlabel('履歴承諾率', fontsize=12)
    ax.set_ylabel('件数', fontsize=12)
    ax.set_title('履歴承諾率の分布', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. IRL vs RF の確率比較（継続者）
    ax = axes[1, 0]
    ax.scatter(continue_df['rf_prob'], continue_df['irl_prob'], 
              alpha=0.6, color='blue', s=100)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('RF確率（継続）', fontsize=12)
    ax.set_ylabel('IRL確率（継続）', fontsize=12)
    ax.set_title(f'継続者の予測確率比較（{len(continue_df)}件）', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # 4. IRL vs RF の確率比較（離脱者）
    ax = axes[1, 1]
    ax.scatter(leave_df['rf_prob'], leave_df['irl_prob'], 
              alpha=0.6, color='red', s=100)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('RF確率（継続）', fontsize=12)
    ax.set_ylabel('IRL確率（継続）', fontsize=12)
    ax.set_title(f'離脱者の予測確率比較（{len(leave_df)}件）', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "irl_only_correct_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n可視化を保存: {output_path}")
    plt.close()


def analyze_specific_developers(df):
    """特定の開発者の詳細分析"""
    print("\n" + "="*80)
    print("【個別開発者の詳細】")
    print("="*80)
    
    # 開発者ごとに集約（複数パターンに登場する可能性がある）
    dev_summary = df.groupby('reviewer_id').agg({
        'true_label': 'first',
        'history_count': 'first',
        'history_rate': 'first',
        'pattern': 'count'  # 何パターンで登場したか
    }).rename(columns={'pattern': 'appearance_count'})
    
    dev_summary = dev_summary.sort_values('appearance_count', ascending=False)
    
    print(f"\n開発者数: {len(dev_summary)}名")
    print(f"複数パターンで登場: {(dev_summary['appearance_count'] > 1).sum()}名")
    
    print("\n最頻出開発者（上位5名）:")
    for idx, (dev_id, row) in enumerate(dev_summary.head(5).iterrows(), 1):
        label_str = "継続者" if row['true_label'] == 1 else "離脱者"
        print(f"{idx}. {dev_id}")
        print(f"   {label_str}, 登場回数={row['appearance_count']}, "
              f"履歴={row['history_count']:.0f}件, 承諾率={row['history_rate']:.3f}")


def main():
    df, continue_df, leave_df = load_and_analyze()
    create_visualizations(df, continue_df, leave_df)
    analyze_specific_developers(df)
    
    print("\n" + "="*80)
    print("分析完了！")
    print("="*80)


if __name__ == "__main__":
    main()

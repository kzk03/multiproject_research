"""
IRL vs RF 10パターン詳細分析

全10パターン（訓練期間 <= 評価期間）でIRLとRFの予測を比較し、
どのような開発者を当てて外しているかを詳細に分析する。
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# パス設定
IRL_DIR = Path("/Users/kazuki-h/research/multiproject_research/results/review_continuation_cross_eval_nova")
RF_DIR = Path("/Users/kazuki-h/research/multiproject_research/outputs/singleproject/rf_nova_case2_simple")
OUTPUT_DIR = Path("/Users/kazuki-h/research/multiproject_research/outputs/irl_rf_10pattern_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 10パターン定義
PATTERNS = [
    ("0-3m", "0-3m"), ("0-3m", "3-6m"), ("0-3m", "6-9m"), ("0-3m", "9-12m"),
    ("3-6m", "3-6m"), ("3-6m", "6-9m"), ("3-6m", "9-12m"),
    ("6-9m", "6-9m"), ("6-9m", "9-12m"),
    ("9-12m", "9-12m")
]


def load_pattern_data(train_period: str, eval_period: str) -> tuple:
    """指定パターンのIRLとRF予測を読み込み"""
    # IRL
    irl_path = IRL_DIR / f"train_{train_period}" / f"eval_{eval_period}" / "predictions.csv"
    irl_df = pd.read_csv(irl_path)
    
    # RF
    rf_path = RF_DIR / f"predictions_{train_period}_to_{eval_period}.csv"
    rf_df = pd.read_csv(rf_path)
    
    return irl_df, rf_df


def merge_and_classify(irl_df: pd.DataFrame, rf_df: pd.DataFrame) -> pd.DataFrame:
    """IRLとRFの予測をマージして分類"""
    # カラム名標準化
    irl_df = irl_df.rename(columns={
        "reviewer_email": "reviewer_id",
        "predicted_binary": "irl_pred",
        "predicted_prob": "irl_prob",
        "true_label": "true_label_irl",
        "history_request_count": "irl_history_count",
        "history_acceptance_rate": "irl_history_rate"
    })
    
    rf_df = rf_df.rename(columns={
        "reviewer_email": "reviewer_id",
        "predicted_label": "rf_pred",
        "predicted_prob": "rf_prob",
        "true_label": "true_label_rf",
        "history_request_count": "rf_history_count",
        "history_acceptance_rate": "rf_history_rate"
    })
    
    # マージ
    merged = pd.merge(
        irl_df[["reviewer_id", "irl_pred", "irl_prob", "true_label_irl", 
                "irl_history_count", "irl_history_rate"]],
        rf_df[["reviewer_id", "rf_pred", "rf_prob", "true_label_rf",
               "rf_history_count", "rf_history_rate"]],
        on="reviewer_id",
        how="inner"
    )
    
    merged["true_label"] = merged["true_label_irl"]
    
    # 分類
    merged["category"] = merged.apply(classify_prediction, axis=1)
    
    return merged


def classify_prediction(row) -> str:
    """予測結果を4カテゴリに分類"""
    irl_correct = row["irl_pred"] == row["true_label"]
    rf_correct = row["rf_pred"] == row["true_label"]
    
    if irl_correct and rf_correct:
        return "both_correct"
    elif not irl_correct and not rf_correct:
        return "both_wrong"
    elif irl_correct and not rf_correct:
        return "irl_only"
    else:
        return "rf_only"


def analyze_developer_characteristics(merged: pd.DataFrame, category: str) -> dict:
    """特定カテゴリの開発者特性を分析"""
    subset = merged[merged["category"] == category]
    
    if len(subset) == 0:
        return {"count": 0}
    
    # 継続者/離脱者の内訳
    continue_count = (subset["true_label"] == 1).sum()
    leave_count = (subset["true_label"] == 0).sum()
    
    # 履歴の統計
    stats = {
        "count": len(subset),
        "continue_count": int(continue_count),
        "leave_count": int(leave_count),
        "avg_history_count": float(subset["irl_history_count"].mean()),
        "median_history_count": float(subset["irl_history_count"].median()),
        "avg_history_rate": float(subset["irl_history_rate"].mean()),
        "median_history_rate": float(subset["irl_history_rate"].median()),
    }
    
    return stats


def print_pattern_analysis(train_p: str, eval_p: str, merged: pd.DataFrame):
    """パターンごとの詳細分析を表示"""
    pattern_name = f"{train_p} → {eval_p}"
    
    print(f"\n{'='*80}")
    print(f"パターン: {pattern_name}")
    print(f"{'='*80}")
    print(f"共通サンプル数: {len(merged)}")
    
    # カテゴリ別集計
    categories = ["both_correct", "both_wrong", "irl_only", "rf_only"]
    
    for cat in categories:
        stats = analyze_developer_characteristics(merged, cat)
        cat_label = {
            "both_correct": "両方正解",
            "both_wrong": "両方不正解", 
            "irl_only": "IRLのみ正解",
            "rf_only": "RFのみ正解"
        }[cat]
        
        if stats["count"] > 0:
            print(f"\n--- {cat_label}: {stats['count']}件 ({stats['count']/len(merged)*100:.1f}%) ---")
            print(f"  継続者: {stats['continue_count']}, 離脱者: {stats['leave_count']}")
            print(f"  履歴依頼数: 平均={stats['avg_history_count']:.1f}, 中央値={stats['median_history_count']:.1f}")
            print(f"  履歴承諾率: 平均={stats['avg_history_rate']:.3f}, 中央値={stats['median_history_rate']:.3f}")


def analyze_all_patterns():
    """全10パターンの分析"""
    all_merged = {}
    pattern_stats = []
    
    # 集約用
    irl_only_all = []
    rf_only_all = []
    both_wrong_all = []
    both_correct_all = []
    
    for train_p, eval_p in PATTERNS:
        irl_df, rf_df = load_pattern_data(train_p, eval_p)
        merged = merge_and_classify(irl_df, rf_df)
        
        pattern_name = f"{train_p}_{eval_p}"
        all_merged[pattern_name] = merged
        
        print_pattern_analysis(train_p, eval_p, merged)
        
        # カテゴリ別に集約
        for _, row in merged.iterrows():
            row_data = {
                "pattern": f"{train_p} → {eval_p}",
                "train_period": train_p,
                "eval_period": eval_p,
                "reviewer_id": row["reviewer_id"],
                "true_label": row["true_label"],
                "irl_pred": row["irl_pred"],
                "rf_pred": row["rf_pred"],
                "irl_prob": row["irl_prob"],
                "rf_prob": row["rf_prob"],
                "history_count": row["irl_history_count"],
                "history_rate": row["irl_history_rate"],
            }
            
            if row["category"] == "irl_only":
                irl_only_all.append(row_data)
            elif row["category"] == "rf_only":
                rf_only_all.append(row_data)
            elif row["category"] == "both_wrong":
                both_wrong_all.append(row_data)
            else:
                both_correct_all.append(row_data)
    
    return irl_only_all, rf_only_all, both_wrong_all, both_correct_all


def analyze_aggregated(irl_only: list, rf_only: list, both_wrong: list, both_correct: list):
    """全パターン集約分析"""
    print("\n" + "="*80)
    print("全10パターン集約分析")
    print("="*80)
    
    total = len(irl_only) + len(rf_only) + len(both_wrong) + len(both_correct)
    
    print(f"\n総サンプル数: {total}")
    print(f"  両方正解: {len(both_correct)} ({len(both_correct)/total*100:.1f}%)")
    print(f"  両方不正解: {len(both_wrong)} ({len(both_wrong)/total*100:.1f}%)")
    print(f"  IRLのみ正解: {len(irl_only)} ({len(irl_only)/total*100:.1f}%)")
    print(f"  RFのみ正解: {len(rf_only)} ({len(rf_only)/total*100:.1f}%)")
    
    # IRLのみ正解の詳細分析
    print("\n" + "-"*60)
    print("【IRLのみ正解】詳細分析")
    print("-"*60)
    
    if len(irl_only) > 0:
        irl_df = pd.DataFrame(irl_only)
        
        # 継続者/離脱者
        continue_df = irl_df[irl_df["true_label"] == 1]
        leave_df = irl_df[irl_df["true_label"] == 0]
        
        print(f"継続者検出: {len(continue_df)}件 ({len(continue_df)/len(irl_df)*100:.1f}%)")
        print(f"離脱者検出: {len(leave_df)}件 ({len(leave_df)/len(irl_df)*100:.1f}%)")
        
        # 履歴特性
        print(f"\n履歴依頼数:")
        print(f"  全体: 平均={irl_df['history_count'].mean():.1f}, 中央値={irl_df['history_count'].median():.1f}")
        if len(continue_df) > 0:
            print(f"  継続者: 平均={continue_df['history_count'].mean():.1f}, 中央値={continue_df['history_count'].median():.1f}")
        if len(leave_df) > 0:
            print(f"  離脱者: 平均={leave_df['history_count'].mean():.1f}, 中央値={leave_df['history_count'].median():.1f}")
        
        print(f"\n履歴承諾率:")
        print(f"  全体: 平均={irl_df['history_rate'].mean():.3f}, 中央値={irl_df['history_rate'].median():.3f}")
        if len(continue_df) > 0:
            print(f"  継続者: 平均={continue_df['history_rate'].mean():.3f}, 中央値={continue_df['history_rate'].median():.3f}")
        if len(leave_df) > 0:
            print(f"  離脱者: 平均={leave_df['history_rate'].mean():.3f}, 中央値={leave_df['history_rate'].median():.3f}")
        
        # パターン別内訳
        print(f"\nパターン別内訳:")
        for pattern, count in irl_df.groupby("pattern").size().sort_values(ascending=False).items():
            sub = irl_df[irl_df["pattern"] == pattern]
            cont = (sub["true_label"] == 1).sum()
            leave = (sub["true_label"] == 0).sum()
            print(f"  {pattern}: {count}件 (継続{cont}/離脱{leave})")
    
    # RFのみ正解の詳細分析
    print("\n" + "-"*60)
    print("【RFのみ正解】詳細分析")
    print("-"*60)
    
    if len(rf_only) > 0:
        rf_df = pd.DataFrame(rf_only)
        
        # 継続者/離脱者
        continue_df = rf_df[rf_df["true_label"] == 1]
        leave_df = rf_df[rf_df["true_label"] == 0]
        
        print(f"継続者検出: {len(continue_df)}件 ({len(continue_df)/len(rf_df)*100:.1f}%)")
        print(f"離脱者検出: {len(leave_df)}件 ({len(leave_df)/len(rf_df)*100:.1f}%)")
        
        # 履歴特性
        print(f"\n履歴依頼数:")
        print(f"  全体: 平均={rf_df['history_count'].mean():.1f}, 中央値={rf_df['history_count'].median():.1f}")
        if len(continue_df) > 0:
            print(f"  継続者: 平均={continue_df['history_count'].mean():.1f}, 中央値={continue_df['history_count'].median():.1f}")
        if len(leave_df) > 0:
            print(f"  離脱者: 平均={leave_df['history_count'].mean():.1f}, 中央値={leave_df['history_count'].median():.1f}")
        
        print(f"\n履歴承諾率:")
        print(f"  全体: 平均={rf_df['history_rate'].mean():.3f}, 中央値={rf_df['history_rate'].median():.3f}")
        if len(continue_df) > 0:
            print(f"  継続者: 平均={continue_df['history_rate'].mean():.3f}, 中央値={continue_df['history_rate'].median():.3f}")
        if len(leave_df) > 0:
            print(f"  離脱者: 平均={leave_df['history_rate'].mean():.3f}, 中央値={leave_df['history_rate'].median():.3f}")
        
        # パターン別内訳
        print(f"\nパターン別内訳:")
        for pattern, count in rf_df.groupby("pattern").size().sort_values(ascending=False).items():
            sub = rf_df[rf_df["pattern"] == pattern]
            cont = (sub["true_label"] == 1).sum()
            leave = (sub["true_label"] == 0).sum()
            print(f"  {pattern}: {count}件 (継続{cont}/離脱{leave})")
    
    # 両方不正解の分析
    print("\n" + "-"*60)
    print("【両方不正解】詳細分析")
    print("-"*60)
    
    if len(both_wrong) > 0:
        bw_df = pd.DataFrame(both_wrong)
        
        continue_df = bw_df[bw_df["true_label"] == 1]
        leave_df = bw_df[bw_df["true_label"] == 0]
        
        print(f"継続者を離脱と誤予測: {len(continue_df)}件 ({len(continue_df)/len(bw_df)*100:.1f}%)")
        print(f"離脱者を継続と誤予測: {len(leave_df)}件 ({len(leave_df)/len(bw_df)*100:.1f}%)")
        
        print(f"\n履歴依頼数:")
        print(f"  全体: 平均={bw_df['history_count'].mean():.1f}, 中央値={bw_df['history_count'].median():.1f}")
        if len(continue_df) > 0:
            print(f"  継続者（誤予測）: 平均={continue_df['history_count'].mean():.1f}, 中央値={continue_df['history_count'].median():.1f}")
        if len(leave_df) > 0:
            print(f"  離脱者（誤予測）: 平均={leave_df['history_count'].mean():.1f}, 中央値={leave_df['history_count'].median():.1f}")
        
        print(f"\n履歴承諾率:")
        print(f"  全体: 平均={bw_df['history_rate'].mean():.3f}, 中央値={bw_df['history_rate'].median():.3f}")
        if len(continue_df) > 0:
            print(f"  継続者（誤予測）: 平均={continue_df['history_rate'].mean():.3f}, 中央値={continue_df['history_rate'].median():.3f}")
        if len(leave_df) > 0:
            print(f"  離脱者（誤予測）: 平均={leave_df['history_rate'].mean():.3f}, 中央値={leave_df['history_rate'].median():.3f}")
    
    return irl_only, rf_only, both_wrong, both_correct


def compare_characteristics():
    """IRLのみ正解 vs RFのみ正解の特性比較"""
    print("\n" + "="*80)
    print("IRLのみ正解 vs RFのみ正解: 特性比較")
    print("="*80)


def save_detailed_results(irl_only: list, rf_only: list, both_wrong: list, both_correct: list):
    """詳細結果をCSVに保存"""
    for name, data in [
        ("irl_only_correct", irl_only),
        ("rf_only_correct", rf_only),
        ("both_wrong", both_wrong),
        ("both_correct", both_correct)
    ]:
        if len(data) > 0:
            df = pd.DataFrame(data)
            df.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)
            print(f"保存: {OUTPUT_DIR / f'{name}.csv'}")


def main():
    print("="*80)
    print("IRL vs RF: 10パターン詳細分析")
    print("="*80)
    
    irl_only, rf_only, both_wrong, both_correct = analyze_all_patterns()
    analyze_aggregated(irl_only, rf_only, both_wrong, both_correct)
    save_detailed_results(irl_only, rf_only, both_wrong, both_correct)
    
    print("\n完了！")


if __name__ == "__main__":
    main()

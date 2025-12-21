"""
IRL vs RF 全期間比較分析

4つの対角線期間（0-3m, 3-6m, 6-9m, 9-12m）で
IRLとRFの予測を個別レベルで比較する。
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# 出力ディレクトリ
OUTPUT_DIR = Path("/Users/kazuki-h/research/multiproject_research/outputs/irl_rf_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# データパス
IRL_DIR = Path("/Users/kazuki-h/research/multiproject_research/results/review_continuation_cross_eval_nova")
RF_DIR = Path("/Users/kazuki-h/research/multiproject_research/outputs/rf_nova_case2_simple")

# 期間マッピング（対角線評価: 訓練期間 = 評価期間）
PERIODS = {
    "0-3m": {"irl_file": "train_0-3m/eval_0-3m/predictions.csv", "rf_file": "predictions_0-3m.csv"},
    "3-6m": {"irl_file": "train_3-6m/eval_3-6m/predictions.csv", "rf_file": "predictions_3-6m.csv"},
    "6-9m": {"irl_file": "train_6-9m/eval_6-9m/predictions.csv", "rf_file": "predictions_6-9m.csv"},
    "9-12m": {"irl_file": "train_9-12m/eval_9-12m/predictions.csv", "rf_file": "predictions_9-12m.csv"},
}


def load_predictions(period: str) -> tuple:
    """指定期間のIRLとRF予測を読み込む"""
    irl_path = IRL_DIR / PERIODS[period]["irl_file"]
    rf_path = RF_DIR / PERIODS[period]["rf_file"]
    
    irl_df = pd.read_csv(irl_path)
    rf_df = pd.read_csv(rf_path)
    
    return irl_df, rf_df


def merge_predictions(irl_df: pd.DataFrame, rf_df: pd.DataFrame) -> pd.DataFrame:
    """IRLとRFの予測をマージ"""
    # カラム名の標準化（IRL: reviewer_email, predicted_binary, predicted_prob, true_label）
    irl_df = irl_df.rename(columns={
        "reviewer_email": "reviewer_id",
        "predicted_binary": "irl_pred",
        "predicted_prob": "irl_prob",
        "true_label": "true_label_irl"
    })
    
    # RF: reviewer_email, predicted_prob, predicted_label, true_label
    rf_df = rf_df.rename(columns={
        "reviewer_email": "reviewer_id",
        "predicted_label": "rf_pred",
        "predicted_prob": "rf_prob",
        "true_label": "true_label_rf"
    })
    
    # reviewer_idでマージ
    merged = pd.merge(
        irl_df[["reviewer_id", "irl_pred", "irl_prob", "true_label_irl"]],
        rf_df[["reviewer_id", "rf_pred", "rf_prob", "true_label_rf"]],
        on="reviewer_id",
        how="inner"
    )
    
    # true_labelの一致確認
    assert (merged["true_label_irl"] == merged["true_label_rf"]).all(), "True labels mismatch!"
    merged["true_label"] = merged["true_label_irl"]
    merged = merged.drop(columns=["true_label_irl", "true_label_rf"])
    
    return merged


def analyze_agreement(merged: pd.DataFrame, period: str) -> dict:
    """予測の一致・不一致を分析"""
    results = {
        "period": period,
        "total_samples": len(merged),
    }
    
    # 一致率
    agree = (merged["irl_pred"] == merged["rf_pred"]).sum()
    results["agreement_rate"] = agree / len(merged)
    
    # 4パターン分類
    both_correct = ((merged["irl_pred"] == merged["true_label"]) & 
                    (merged["rf_pred"] == merged["true_label"])).sum()
    both_wrong = ((merged["irl_pred"] != merged["true_label"]) & 
                  (merged["rf_pred"] != merged["true_label"])).sum()
    irl_only_correct = ((merged["irl_pred"] == merged["true_label"]) & 
                        (merged["rf_pred"] != merged["true_label"])).sum()
    rf_only_correct = ((merged["irl_pred"] != merged["true_label"]) & 
                       (merged["rf_pred"] == merged["true_label"])).sum()
    
    results["both_correct"] = both_correct
    results["both_wrong"] = both_wrong
    results["irl_only_correct"] = irl_only_correct
    results["rf_only_correct"] = rf_only_correct
    
    # パーセンテージ
    total = len(merged)
    results["both_correct_pct"] = both_correct / total * 100
    results["both_wrong_pct"] = both_wrong / total * 100
    results["irl_only_correct_pct"] = irl_only_correct / total * 100
    results["rf_only_correct_pct"] = rf_only_correct / total * 100
    
    # 各手法の指標
    for name, pred_col in [("irl", "irl_pred"), ("rf", "rf_pred")]:
        results[f"{name}_f1"] = f1_score(merged["true_label"], merged[pred_col])
        results[f"{name}_recall"] = recall_score(merged["true_label"], merged[pred_col])
        results[f"{name}_precision"] = precision_score(merged["true_label"], merged[pred_col])
        prob_col = f"{name}_prob"
        results[f"{name}_auc"] = roc_auc_score(merged["true_label"], merged[prob_col])
    
    return results


def create_comparison_heatmap(all_results: list):
    """全期間の比較ヒートマップを作成"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    periods = ["0-3m", "3-6m", "6-9m", "9-12m"]
    
    for idx, (ax, period) in enumerate(zip(axes.flatten(), periods)):
        result = all_results[idx]
        
        # 4パターンのマトリックス
        data = np.array([
            [result["both_correct"], result["rf_only_correct"]],
            [result["irl_only_correct"], result["both_wrong"]]
        ])
        
        labels = np.array([
            [f"Both Correct\n{result['both_correct']} ({result['both_correct_pct']:.1f}%)",
             f"RF Only\n{result['rf_only_correct']} ({result['rf_only_correct_pct']:.1f}%)"],
            [f"IRL Only\n{result['irl_only_correct']} ({result['irl_only_correct_pct']:.1f}%)",
             f"Both Wrong\n{result['both_wrong']} ({result['both_wrong_pct']:.1f}%)"]
        ])
        
        sns.heatmap(data, annot=labels, fmt="", cmap="RdYlGn", ax=ax,
                    xticklabels=["RF Correct", "RF Wrong"],
                    yticklabels=["IRL Correct", "IRL Wrong"],
                    cbar=True, vmin=0, vmax=result["total_samples"]*0.6)
        
        ax.set_title(f"{period}\nIRL F1={result['irl_f1']:.3f}, RF F1={result['rf_f1']:.3f}\n"
                     f"Agreement: {result['agreement_rate']:.1%}", fontsize=11)
    
    plt.suptitle("IRL vs RF Case2: Prediction Agreement by Period\n(Nova Project, Cross-Temporal)", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "irl_rf_agreement_all_periods.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: {OUTPUT_DIR / 'irl_rf_agreement_all_periods.png'}")


def create_metrics_comparison_plot(all_results: list):
    """指標比較のバープロット"""
    periods = ["0-3m", "3-6m", "6-9m", "9-12m"]
    metrics = ["f1", "auc", "recall", "precision"]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for ax, metric in zip(axes.flatten(), metrics):
        irl_vals = [r[f"irl_{metric}"] for r in all_results]
        rf_vals = [r[f"rf_{metric}"] for r in all_results]
        
        x = np.arange(len(periods))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, irl_vals, width, label='IRL', color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, rf_vals, width, label='RF Case2', color='coral', alpha=0.8)
        
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(periods)
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        
        # 値ラベル
        for bar in bars1:
            ax.annotate(f'{bar.get_height():.2f}',
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            ax.annotate(f'{bar.get_height():.2f}',
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=9)
    
    plt.suptitle("IRL vs RF Case2: Metrics by Period\n(Nova Project, Cross-Temporal)", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "irl_rf_metrics_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: {OUTPUT_DIR / 'irl_rf_metrics_comparison.png'}")


def create_disagreement_analysis(all_merged: dict):
    """不一致ケースの詳細分析"""
    print("\n" + "="*80)
    print("不一致ケースの詳細分析")
    print("="*80)
    
    for period, merged in all_merged.items():
        print(f"\n--- {period} ---")
        
        # IRLのみ正解
        irl_only = merged[(merged["irl_pred"] == merged["true_label"]) & 
                          (merged["rf_pred"] != merged["true_label"])]
        
        # RFのみ正解
        rf_only = merged[(merged["irl_pred"] != merged["true_label"]) & 
                         (merged["rf_pred"] == merged["true_label"])]
        
        print(f"IRLのみ正解: {len(irl_only)} 件")
        if len(irl_only) > 0:
            # 継続予測の正解率
            irl_only_continue = irl_only[irl_only["true_label"] == 1]
            irl_only_leave = irl_only[irl_only["true_label"] == 0]
            print(f"  - 継続者の検出: {len(irl_only_continue)} 件")
            print(f"  - 離脱者の検出: {len(irl_only_leave)} 件")
        
        print(f"RFのみ正解: {len(rf_only)} 件")
        if len(rf_only) > 0:
            rf_only_continue = rf_only[rf_only["true_label"] == 1]
            rf_only_leave = rf_only[rf_only["true_label"] == 0]
            print(f"  - 継続者の検出: {len(rf_only_continue)} 件")
            print(f"  - 離脱者の検出: {len(rf_only_leave)} 件")


def main():
    print("="*80)
    print("IRL vs RF Case2: 全期間比較分析")
    print("="*80)
    
    all_results = []
    all_merged = {}
    
    for period in ["0-3m", "3-6m", "6-9m", "9-12m"]:
        print(f"\n--- {period} ---")
        
        # データ読み込み
        irl_df, rf_df = load_predictions(period)
        print(f"IRL予測数: {len(irl_df)}, RF予測数: {len(rf_df)}")
        
        # マージ
        merged = merge_predictions(irl_df, rf_df)
        print(f"共通サンプル数: {len(merged)}")
        all_merged[period] = merged
        
        # 分析
        result = analyze_agreement(merged, period)
        all_results.append(result)
        
        # 結果表示
        print(f"一致率: {result['agreement_rate']:.1%}")
        print(f"IRL F1: {result['irl_f1']:.3f}, RF F1: {result['rf_f1']:.3f}")
        print(f"IRL AUC: {result['irl_auc']:.3f}, RF AUC: {result['rf_auc']:.3f}")
        print(f"両方正解: {result['both_correct']} ({result['both_correct_pct']:.1f}%)")
        print(f"両方不正解: {result['both_wrong']} ({result['both_wrong_pct']:.1f}%)")
        print(f"IRLのみ正解: {result['irl_only_correct']} ({result['irl_only_correct_pct']:.1f}%)")
        print(f"RFのみ正解: {result['rf_only_correct']} ({result['rf_only_correct_pct']:.1f}%)")
    
    # 可視化
    print("\n" + "="*80)
    print("可視化生成中...")
    create_comparison_heatmap(all_results)
    create_metrics_comparison_plot(all_results)
    
    # 不一致分析
    create_disagreement_analysis(all_merged)
    
    # サマリー
    print("\n" + "="*80)
    print("全期間サマリー")
    print("="*80)
    
    avg_irl_f1 = np.mean([r["irl_f1"] for r in all_results])
    avg_rf_f1 = np.mean([r["rf_f1"] for r in all_results])
    avg_irl_auc = np.mean([r["irl_auc"] for r in all_results])
    avg_rf_auc = np.mean([r["rf_auc"] for r in all_results])
    avg_agreement = np.mean([r["agreement_rate"] for r in all_results])
    
    total_irl_only = sum(r["irl_only_correct"] for r in all_results)
    total_rf_only = sum(r["rf_only_correct"] for r in all_results)
    total_both_correct = sum(r["both_correct"] for r in all_results)
    total_both_wrong = sum(r["both_wrong"] for r in all_results)
    total_samples = sum(r["total_samples"] for r in all_results)
    
    print(f"平均 IRL F1: {avg_irl_f1:.3f}")
    print(f"平均 RF F1:  {avg_rf_f1:.3f}")
    print(f"平均 IRL AUC: {avg_irl_auc:.3f}")
    print(f"平均 RF AUC:  {avg_rf_auc:.3f}")
    print(f"平均一致率: {avg_agreement:.1%}")
    print(f"\n全期間合計:")
    print(f"  両方正解: {total_both_correct} ({total_both_correct/total_samples*100:.1f}%)")
    print(f"  両方不正解: {total_both_wrong} ({total_both_wrong/total_samples*100:.1f}%)")
    print(f"  IRLのみ正解: {total_irl_only} ({total_irl_only/total_samples*100:.1f}%)")
    print(f"  RFのみ正解: {total_rf_only} ({total_rf_only/total_samples*100:.1f}%)")
    
    # 結果保存
    def convert_to_serializable(obj):
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj
    
    summary = {
        "periods": [convert_to_serializable(r) for r in all_results],
        "summary": convert_to_serializable({
            "avg_irl_f1": avg_irl_f1,
            "avg_rf_f1": avg_rf_f1,
            "avg_irl_auc": avg_irl_auc,
            "avg_rf_auc": avg_rf_auc,
            "avg_agreement": avg_agreement,
            "total_samples": total_samples,
            "total_irl_only_correct": total_irl_only,
            "total_rf_only_correct": total_rf_only,
            "total_both_correct": total_both_correct,
            "total_both_wrong": total_both_wrong,
        })
    }
    
    with open(OUTPUT_DIR / "comparison_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n結果保存: {OUTPUT_DIR / 'comparison_results.json'}")
    
    # マージデータ保存
    for period, merged in all_merged.items():
        merged.to_csv(OUTPUT_DIR / f"merged_{period}.csv", index=False)
    print(f"マージデータ保存: {OUTPUT_DIR}/merged_*.csv")


if __name__ == "__main__":
    main()

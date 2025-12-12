import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from scipy import stats

# スタイル設定
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_all_predictions(results_dir: Path) -> Dict[str, pd.DataFrame]:
    """全ての予測結果を読み込む"""
    experiments = {
        'Single Project (Nova)': results_dir / "review_continuation_cross_eval_nova",
        'Multi-Project (No OS)': results_dir / "cross_temporal_openstack_multiproject_2020_2024",
        'Multi-Project (2x OS)': results_dir / "cross_temporal_openstack_multiproject_os2x",
        'Multi-Project (3x OS)': results_dir / "cross_temporal_openstack_multiproject_os3x",
    }

    train_periods = ['0-3m', '3-6m', '6-9m', '9-12m']
    eval_periods = ['0-3m', '3-6m', '6-9m', '9-12m']

    all_predictions = {}

    for exp_name, path in experiments.items():
        if not path.exists():
            continue

        exp_predictions = []
        for train in train_periods:
            for eval_p in eval_periods:
                pred_file = path / f"train_{train}" / f"eval_{eval_p}" / "predictions.csv"
                if pred_file.exists():
                    df = pd.read_csv(pred_file)
                    df['train_period'] = train
                    df['eval_period'] = eval_p
                    df['experiment'] = exp_name
                    exp_predictions.append(df)

        if exp_predictions:
            all_predictions[exp_name] = pd.concat(exp_predictions, ignore_index=True)

    return all_predictions

def categorize_reviewers(df: pd.DataFrame) -> pd.DataFrame:
    """レビュアーを属性でカテゴリ化"""
    df = df.copy()

    # 経験レベル（履歴レビュー数）
    df['experience_level'] = pd.cut(
        df['history_request_count'],
        bins=[-1, 10, 50, 150, float('inf')],
        labels=['Novice (≤10)', 'Intermediate (11-50)', 'Advanced (51-150)', 'Expert (>150)']
    )

    # 過去の承諾率
    df['past_acceptance_category'] = pd.cut(
        df['history_acceptance_rate'],
        bins=[-0.01, 0.3, 0.5, 0.7, 1.01],
        labels=['Low (<30%)', 'Medium (30-50%)', 'High (50-70%)', 'Very High (>70%)']
    )

    # 活動レベル（評価期間のレビュー数）
    df['activity_level'] = pd.cut(
        df['eval_request_count'],
        bins=[-1, 5, 20, 50, float('inf')],
        labels=['Inactive (≤5)', 'Moderate (6-20)', 'Active (21-50)', 'Very Active (>50)']
    )

    # 実際の承諾率（評価期間）
    df['actual_acceptance_rate'] = df['eval_accepted_count'] / df['eval_request_count']

    return df

def analyze_prediction_by_attribute(df: pd.DataFrame, attribute: str, exp_name: str) -> pd.DataFrame:
    """属性別の予測精度を分析"""
    results = []

    for category in df[attribute].cat.categories:
        subset = df[df[attribute] == category]

        if len(subset) == 0:
            continue

        # メトリクスを計算
        y_true = subset['true_label'].values
        y_pred_binary = subset['predicted_binary'].values
        y_pred_prob = subset['predicted_prob'].values

        tp = ((y_pred_binary == 1) & (y_true == 1)).sum()
        fp = ((y_pred_binary == 1) & (y_true == 0)).sum()
        tn = ((y_pred_binary == 0) & (y_true == 0)).sum()
        fn = ((y_pred_binary == 0) & (y_true == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results.append({
            'experiment': exp_name,
            'attribute': attribute,
            'category': category,
            'count': len(subset),
            'positive_count': y_true.sum(),
            'positive_rate': y_true.mean(),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'mean_predicted_prob': y_pred_prob.mean(),
            'std_predicted_prob': y_pred_prob.std(),
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        })

    return pd.DataFrame(results)

def plot_performance_by_experience(all_predictions: Dict[str, pd.DataFrame], output_dir: Path):
    """経験レベル別の性能を可視化"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Prediction Performance by Reviewer Experience Level', fontsize=16, fontweight='bold')

    metrics = ['precision', 'recall', 'f1_score', 'mean_predicted_prob']
    metric_names = ['Precision', 'Recall', 'F1 Score', 'Mean Predicted Prob']

    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 2, idx % 2]

        # 各実験設定のデータを準備
        for exp_name, df in all_predictions.items():
            df_cat = categorize_reviewers(df)
            analysis = analyze_prediction_by_attribute(df_cat, 'experience_level', exp_name)

            categories = analysis['category'].values
            values = analysis[metric].values

            ax.plot(range(len(categories)), values, marker='o', label=exp_name.replace('Multi-Project', 'MP'),
                   linewidth=2, markersize=8)

        ax.set_xlabel('Experience Level', fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(f'{metric_name} by Experience Level', fontsize=13, fontweight='bold')
        ax.set_xticks(range(4))
        ax.set_xticklabels(['Novice\n(≤10)', 'Intermediate\n(11-50)', 'Advanced\n(51-150)', 'Expert\n(>150)'])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_by_experience.png', dpi=300, bbox_inches='tight')
    print(f"保存しました: {output_dir / 'performance_by_experience.png'}")

def plot_performance_by_past_acceptance(all_predictions: Dict[str, pd.DataFrame], output_dir: Path):
    """過去の承諾率別の性能を可視化"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Prediction Performance by Past Acceptance Rate', fontsize=16, fontweight='bold')

    metrics = ['precision', 'recall', 'f1_score', 'positive_rate']
    metric_names = ['Precision', 'Recall', 'F1 Score', 'Actual Acceptance Rate']

    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 2, idx % 2]

        for exp_name, df in all_predictions.items():
            df_cat = categorize_reviewers(df)
            analysis = analyze_prediction_by_attribute(df_cat, 'past_acceptance_category', exp_name)

            categories = analysis['category'].values
            values = analysis[metric].values

            ax.plot(range(len(categories)), values, marker='s', label=exp_name.replace('Multi-Project', 'MP'),
                   linewidth=2, markersize=8)

        ax.set_xlabel('Past Acceptance Rate', fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(f'{metric_name} by Past Acceptance Rate', fontsize=13, fontweight='bold')
        ax.set_xticks(range(4))
        ax.set_xticklabels(['Low\n(<30%)', 'Medium\n(30-50%)', 'High\n(50-70%)', 'Very High\n(>70%)'])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_by_past_acceptance.png', dpi=300, bbox_inches='tight')
    print(f"保存しました: {output_dir / 'performance_by_past_acceptance.png'}")

def plot_performance_by_activity(all_predictions: Dict[str, pd.DataFrame], output_dir: Path):
    """活動レベル別の性能を可視化"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Prediction Performance by Activity Level', fontsize=16, fontweight='bold')

    metrics = ['precision', 'recall', 'f1_score', 'count']
    metric_names = ['Precision', 'Recall', 'F1 Score', 'Sample Count']

    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 2, idx % 2]

        for exp_name, df in all_predictions.items():
            df_cat = categorize_reviewers(df)
            analysis = analyze_prediction_by_attribute(df_cat, 'activity_level', exp_name)

            categories = analysis['category'].values
            values = analysis[metric].values

            ax.plot(range(len(categories)), values, marker='^', label=exp_name.replace('Multi-Project', 'MP'),
                   linewidth=2, markersize=8)

        ax.set_xlabel('Activity Level', fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(f'{metric_name} by Activity Level', fontsize=13, fontweight='bold')
        ax.set_xticks(range(4))
        ax.set_xticklabels(['Inactive\n(≤5)', 'Moderate\n(6-20)', 'Active\n(21-50)', 'Very Active\n(>50)'])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        if metric != 'count':
            ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_by_activity.png', dpi=300, bbox_inches='tight')
    print(f"保存しました: {output_dir / 'performance_by_activity.png'}")

def plot_calibration_by_attribute(all_predictions: Dict[str, pd.DataFrame], output_dir: Path):
    """属性別のキャリブレーション分析"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Prediction Calibration by Reviewer Attributes', fontsize=16, fontweight='bold')

    # 各実験設定について分析
    for idx, (exp_name, df) in enumerate(all_predictions.items()):
        ax = axes[idx // 2, idx % 2]
        df_cat = categorize_reviewers(df)

        # 経験レベル別のキャリブレーション
        for experience in df_cat['experience_level'].cat.categories:
            subset = df_cat[df_cat['experience_level'] == experience]
            if len(subset) == 0:
                continue

            # 予測確率を10個のビンに分割
            bins = np.linspace(0, 1, 11)
            bin_indices = np.digitize(subset['predicted_prob'], bins)

            bin_true_rates = []
            bin_pred_rates = []
            bin_centers = []

            for i in range(1, len(bins)):
                mask = bin_indices == i
                if mask.sum() > 0:
                    bin_true_rates.append(subset[mask]['true_label'].mean())
                    bin_pred_rates.append(subset[mask]['predicted_prob'].mean())
                    bin_centers.append((bins[i-1] + bins[i]) / 2)

            if bin_pred_rates:
                ax.plot(bin_pred_rates, bin_true_rates, marker='o',
                       label=f'{experience}', linewidth=2, markersize=6)

        # 完全なキャリブレーション線
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)

        ax.set_xlabel('Predicted Probability', fontsize=11)
        ax.set_ylabel('Actual Acceptance Rate', fontsize=11)
        ax.set_title(f'{exp_name.replace("Multi-Project", "MP")}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_dir / 'calibration_by_experience.png', dpi=300, bbox_inches='tight')
    print(f"保存しました: {output_dir / 'calibration_by_experience.png'}")

def create_summary_table(all_predictions: Dict[str, pd.DataFrame], output_dir: Path):
    """属性別のサマリーテーブルを作成"""
    print("\n" + "="*100)
    print("レビュアー属性別の予測性能分析")
    print("="*100)

    for exp_name, df in all_predictions.items():
        df_cat = categorize_reviewers(df)

        print(f"\n【{exp_name}】")
        print("-"*100)

        # 経験レベル別
        print("\n1. 経験レベル別の性能:")
        print("-"*100)
        analysis = analyze_prediction_by_attribute(df_cat, 'experience_level', exp_name)
        print(analysis[['category', 'count', 'positive_rate', 'precision', 'recall', 'f1_score']].to_string(index=False))

        # 過去の承諾率別
        print("\n2. 過去の承諾率別の性能:")
        print("-"*100)
        analysis = analyze_prediction_by_attribute(df_cat, 'past_acceptance_category', exp_name)
        print(analysis[['category', 'count', 'positive_rate', 'precision', 'recall', 'f1_score']].to_string(index=False))

        # 活動レベル別
        print("\n3. 活動レベル別の性能:")
        print("-"*100)
        analysis = analyze_prediction_by_attribute(df_cat, 'activity_level', exp_name)
        print(analysis[['category', 'count', 'positive_rate', 'precision', 'recall', 'f1_score']].to_string(index=False))

        # CSVに保存
        exp_output_dir = output_dir / exp_name.replace(' ', '_').replace('(', '').replace(')', '')
        exp_output_dir.mkdir(parents=True, exist_ok=True)

        for attr in ['experience_level', 'past_acceptance_category', 'activity_level']:
            analysis = analyze_prediction_by_attribute(df_cat, attr, exp_name)
            analysis.to_csv(exp_output_dir / f'analysis_by_{attr}.csv', index=False)

def analyze_error_patterns(all_predictions: Dict[str, pd.DataFrame], output_dir: Path):
    """エラーパターンの分析"""
    print("\n" + "="*100)
    print("エラーパターンの詳細分析")
    print("="*100)

    for exp_name, df in all_predictions.items():
        df_cat = categorize_reviewers(df)

        print(f"\n【{exp_name}】")
        print("-"*100)

        # False Positive（承諾すると予測したが実際は拒否）
        fp_df = df_cat[(df_cat['predicted_binary'] == 1) & (df_cat['true_label'] == 0)]
        print(f"\nFalse Positive数: {len(fp_df)}")
        if len(fp_df) > 0:
            print("False Positiveの特徴:")
            print(f"  平均経験: {fp_df['history_request_count'].mean():.1f} レビュー")
            print(f"  平均過去承諾率: {fp_df['history_acceptance_rate'].mean():.3f}")
            print(f"  平均予測確率: {fp_df['predicted_prob'].mean():.3f}")
            print(f"  経験レベル分布:")
            print(fp_df['experience_level'].value_counts().to_string())

        # False Negative（拒否すると予測したが実際は承諾）
        fn_df = df_cat[(df_cat['predicted_binary'] == 0) & (df_cat['true_label'] == 1)]
        print(f"\nFalse Negative数: {len(fn_df)}")
        if len(fn_df) > 0:
            print("False Negativeの特徴:")
            print(f"  平均経験: {fn_df['history_request_count'].mean():.1f} レビュー")
            print(f"  平均過去承諾率: {fn_df['history_acceptance_rate'].mean():.3f}")
            print(f"  平均予測確率: {fn_df['predicted_prob'].mean():.3f}")
            print(f"  経験レベル分布:")
            print(fn_df['experience_level'].value_counts().to_string())

def main():
    results_dir = Path("/Users/kazuki-h/research/multiproject_research/results")
    output_dir = Path("/Users/kazuki-h/research/multiproject_research/docs/figures/reviewer_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("予測結果を読み込み中...")
    all_predictions = load_all_predictions(results_dir)

    print(f"\n読み込まれた実験: {list(all_predictions.keys())}")
    for exp_name, df in all_predictions.items():
        print(f"  {exp_name}: {len(df)} 予測")

    print("\n可視化を生成中...")

    print("\n1. 経験レベル別の性能...")
    plot_performance_by_experience(all_predictions, output_dir)

    print("\n2. 過去の承諾率別の性能...")
    plot_performance_by_past_acceptance(all_predictions, output_dir)

    print("\n3. 活動レベル別の性能...")
    plot_performance_by_activity(all_predictions, output_dir)

    print("\n4. キャリブレーション分析...")
    plot_calibration_by_attribute(all_predictions, output_dir)

    print("\n5. サマリーテーブル作成...")
    create_summary_table(all_predictions, output_dir)

    print("\n6. エラーパターン分析...")
    analyze_error_patterns(all_predictions, output_dir)

    print(f"\n全ての分析が完了しました: {output_dir}")

if __name__ == "__main__":
    main()

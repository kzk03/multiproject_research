"""
各プロジェクトのIRLモデルの予測精度を比較するスクリプト
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
import glob
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, precision_score, recall_score

def load_metrics_from_json(project_path, project_name):
    """metrics.jsonファイルから精度指標を読み込む"""
    results = []

    # すべてのmetrics.jsonファイルを探す
    json_files = glob.glob(str(project_path / "**" / "metrics.json"), recursive=True)

    for json_file in json_files:
        # パスから学習期間と評価期間を抽出
        path_parts = Path(json_file).parts
        train_period = None
        eval_period = None

        for i, part in enumerate(path_parts):
            if part.startswith('train_'):
                train_period = part.replace('train_', '')
            if part.startswith('eval_'):
                eval_period = part.replace('eval_', '')

        if not train_period or not eval_period:
            continue

        # metrics.jsonを読み込み
        with open(json_file, 'r') as f:
            metrics = json.load(f)

        metrics['project'] = project_name
        metrics['train_period'] = train_period
        metrics['eval_period'] = eval_period

        results.append(metrics)

    return pd.DataFrame(results) if results else pd.DataFrame()


def calculate_metrics_from_predictions(project_path, project_name):
    """predictions.csvから精度指標を計算"""
    results = []

    # すべてのpredictions.csvファイルを探す
    csv_files = glob.glob(str(project_path / "**" / "predictions.csv"), recursive=True)

    for csv_file in csv_files:
        # パスから学習期間と評価期間を抽出
        path_parts = Path(csv_file).parts
        train_period = None
        eval_period = None

        for i, part in enumerate(path_parts):
            if part.startswith('train_'):
                train_period = part.replace('train_', '')
            if part.startswith('eval_'):
                eval_period = part.replace('eval_', '')

        if not train_period or not eval_period:
            continue

        # データを読み込み
        df = pd.read_csv(csv_file)

        if len(df) == 0 or 'true_label' not in df.columns or 'predicted_prob' not in df.columns:
            continue

        y_true = df['true_label']
        y_pred_prob = df['predicted_prob']
        y_pred_binary = df['predicted_binary'] if 'predicted_binary' in df.columns else (y_pred_prob > 0.5).astype(int)

        # 精度指標を計算
        try:
            auc_roc = roc_auc_score(y_true, y_pred_prob)
        except:
            auc_roc = np.nan

        try:
            precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_prob)
            auc_pr = auc(recall_vals, precision_vals)
        except:
            auc_pr = np.nan

        try:
            f1 = f1_score(y_true, y_pred_binary)
        except:
            f1 = np.nan

        try:
            precision = precision_score(y_true, y_pred_binary)
        except:
            precision = np.nan

        try:
            recall = recall_score(y_true, y_pred_binary)
        except:
            recall = np.nan

        results.append({
            'project': project_name,
            'train_period': train_period,
            'eval_period': eval_period,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'total_samples': len(df),
            'positive_samples': y_true.sum(),
            'negative_samples': len(df) - y_true.sum()
        })

    return pd.DataFrame(results) if results else pd.DataFrame()


def main():
    base_path = Path("/Users/kazuki-h/research/multiproject_research/outputs/multiprojects")

    projects = {
        'Qt': base_path / "qt_50projects_irl",
        'Android': base_path / "android_50projects_irl",
        'Chromium': base_path / "chromium_50projects_irl",
        'OpenStack': base_path / "opnestack_50projects_irl_timeseries" / "2x_os",
    }

    all_metrics = []

    for project_name, project_path in projects.items():
        print(f"\n{'='*60}")
        print(f"分析中: {project_name}")
        print(f"{'='*60}")

        if not project_path.exists():
            print(f"警告: {project_path} が見つかりません")
            continue

        # metrics.jsonから読み込み
        metrics_df = load_metrics_from_json(project_path, project_name)

        if len(metrics_df) > 0:
            all_metrics.append(metrics_df)

            # プロジェクトごとの統計
            print(f"\n{project_name} のモデル精度統計:")
            print(f"  データポイント数: {len(metrics_df)}")
            print(f"  平均 AUC-ROC: {metrics_df['auc_roc'].mean():.4f} ± {metrics_df['auc_roc'].std():.4f}")
            print(f"  平均 AUC-PR: {metrics_df['auc_pr'].mean():.4f} ± {metrics_df['auc_pr'].std():.4f}")
            print(f"  平均 F1 Score: {metrics_df['f1_score'].mean():.4f} ± {metrics_df['f1_score'].std():.4f}")
            print(f"  平均 Precision: {metrics_df['precision'].mean():.4f} ± {metrics_df['precision'].std():.4f}")
            print(f"  平均 Recall: {metrics_df['recall'].mean():.4f} ± {metrics_df['recall'].std():.4f}")

    # 全プロジェクトの結果を結合
    if all_metrics:
        combined_df = pd.concat(all_metrics, ignore_index=True)

        # 結果を保存
        output_dir = Path("/Users/kazuki-h/research/multiproject_research/results/irl_model_performance")
        output_dir.mkdir(parents=True, exist_ok=True)

        combined_df.to_csv(output_dir / "all_projects_model_performance.csv", index=False)
        print(f"\n\n{'='*60}")
        print("結果を保存しました:")
        print(f"  {output_dir / 'all_projects_model_performance.csv'}")

        # プロジェクト別サマリー
        print(f"\n{'='*60}")
        print("プロジェクト別モデル性能サマリー")
        print(f"{'='*60}")

        summary = combined_df.groupby('project').agg({
            'auc_roc': ['mean', 'std', 'min', 'max'],
            'auc_pr': ['mean', 'std', 'min', 'max'],
            'f1_score': ['mean', 'std', 'min', 'max'],
            'precision': ['mean', 'std', 'min', 'max'],
            'recall': ['mean', 'std', 'min', 'max'],
            'total_count': 'sum'
        }).round(4)

        print(summary)

        # より見やすい形式でサマリーを作成
        simple_summary = combined_df.groupby('project').agg({
            'auc_roc': 'mean',
            'auc_pr': 'mean',
            'f1_score': 'mean',
            'precision': 'mean',
            'recall': 'mean',
            'total_count': 'mean'
        }).round(4)

        simple_summary = simple_summary.sort_values('auc_roc', ascending=False)

        print(f"\n{'='*60}")
        print("プロジェクト別平均性能（AUC-ROCでソート）")
        print(f"{'='*60}")
        print(simple_summary)

        simple_summary.to_csv(output_dir / "project_performance_summary.csv")
        print(f"\nサマリーを保存しました: {output_dir / 'project_performance_summary.csv'}")

        # 各プロジェクトの詳細を個別ファイルに保存
        for project_name in combined_df['project'].unique():
            project_df = combined_df[combined_df['project'] == project_name]
            project_df.to_csv(output_dir / f"{project_name.lower()}_performance.csv", index=False)
            print(f"  {output_dir / f'{project_name.lower()}_performance.csv'}")

        # 評価期間ごとの性能比較
        print(f"\n{'='*60}")
        print("評価期間ごとの平均性能")
        print(f"{'='*60}")

        eval_period_summary = combined_df.groupby(['project', 'eval_period']).agg({
            'auc_roc': 'mean',
            'f1_score': 'mean',
            'precision': 'mean',
            'recall': 'mean'
        }).round(4)

        print(eval_period_summary)
        eval_period_summary.to_csv(output_dir / "eval_period_performance.csv")

        # 学習期間ごとの性能比較
        print(f"\n{'='*60}")
        print("学習期間ごとの平均性能")
        print(f"{'='*60}")

        train_period_summary = combined_df.groupby(['project', 'train_period']).agg({
            'auc_roc': 'mean',
            'f1_score': 'mean',
            'precision': 'mean',
            'recall': 'mean'
        }).round(4)

        print(train_period_summary)
        train_period_summary.to_csv(output_dir / "train_period_performance.csv")


if __name__ == "__main__":
    main()

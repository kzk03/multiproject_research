import json
import os
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple

def load_metrics(base_path: str, train_period: str, eval_period: str) -> Dict:
    """メトリクスファイルを読み込む"""
    metrics_path = Path(base_path) / f"train_{train_period}" / f"eval_{eval_period}" / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None

def load_optimal_threshold(base_path: str, train_period: str) -> Dict:
    """最適閾値ファイルを読み込む"""
    threshold_path = Path(base_path) / f"train_{train_period}" / "optimal_threshold.json"
    if threshold_path.exists():
        with open(threshold_path, 'r') as f:
            return json.load(f)
    return None

def calculate_statistics(metrics_list: List[Dict]) -> Dict:
    """メトリクスの統計を計算"""
    if not metrics_list:
        return {}

    keys = ['precision', 'recall', 'f1_score', 'auc_roc', 'auc_pr']
    stats = {}

    for key in keys:
        values = [m[key] for m in metrics_list if m and key in m]
        if values:
            stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }

    return stats

def analyze_cross_temporal_performance(base_path: str, name: str) -> Dict:
    """クロステンポラル評価の分析"""
    train_periods = ['0-3m', '3-6m', '6-9m', '9-12m']
    eval_periods = ['0-3m', '3-6m', '6-9m', '9-12m']

    results = {
        'name': name,
        'same_period': [],  # 同一期間での評価
        'future_period': [],  # 未来期間での評価
        'all_evaluations': []
    }

    for train_period in train_periods:
        for eval_period in eval_periods:
            metrics = load_metrics(base_path, train_period, eval_period)
            if metrics:
                results['all_evaluations'].append({
                    'train': train_period,
                    'eval': eval_period,
                    'metrics': metrics
                })

                # 同一期間
                if train_period == eval_period:
                    results['same_period'].append(metrics)
                # 未来期間（訓練期間より後の期間での評価）
                elif train_periods.index(eval_period) > train_periods.index(train_period):
                    results['future_period'].append(metrics)

    # 統計計算
    results['same_period_stats'] = calculate_statistics(results['same_period'])
    results['future_period_stats'] = calculate_statistics(results['future_period'])

    return results

def print_comparison_table(results_dict: Dict[str, Dict]):
    """比較表を出力"""
    print("\n" + "="*100)
    print("実験結果の詳細比較分析")
    print("="*100)

    # 1. 同一期間での性能比較
    print("\n【1. 同一期間（Same Period）での性能】")
    print("-"*100)
    print(f"{'実験設定':<40} {'F1 (mean±std)':<20} {'Precision':<20} {'Recall':<20} {'AUC-ROC':<15}")
    print("-"*100)

    for name, results in results_dict.items():
        stats = results['same_period_stats']
        if 'f1_score' in stats:
            f1 = f"{stats['f1_score']['mean']:.4f}±{stats['f1_score']['std']:.4f}"
            prec = f"{stats['precision']['mean']:.4f}±{stats['precision']['std']:.4f}"
            rec = f"{stats['recall']['mean']:.4f}±{stats['recall']['std']:.4f}"
            auc = f"{stats['auc_roc']['mean']:.4f}±{stats['auc_roc']['std']:.4f}"
            print(f"{name:<40} {f1:<20} {prec:<20} {rec:<20} {auc:<15}")

    # 2. 未来期間での性能比較（汎化性能）
    print("\n【2. 未来期間（Future Period）での性能（汎化性能）】")
    print("-"*100)
    print(f"{'実験設定':<40} {'F1 (mean±std)':<20} {'Precision':<20} {'Recall':<20} {'AUC-ROC':<15}")
    print("-"*100)

    for name, results in results_dict.items():
        stats = results['future_period_stats']
        if 'f1_score' in stats:
            f1 = f"{stats['f1_score']['mean']:.4f}±{stats['f1_score']['std']:.4f}"
            prec = f"{stats['precision']['mean']:.4f}±{stats['precision']['std']:.4f}"
            rec = f"{stats['recall']['mean']:.4f}±{stats['recall']['std']:.4f}"
            auc = f"{stats['auc_roc']['mean']:.4f}±{stats['auc_roc']['std']:.4f}"
            print(f"{name:<40} {f1:<20} {prec:<20} {rec:<20} {auc:<15}")

    # 3. 性能低下の分析
    print("\n【3. 性能低下分析（同一期間 vs 未来期間）】")
    print("-"*100)
    print(f"{'実験設定':<40} {'F1低下':<15} {'Precision低下':<15} {'Recall低下':<15} {'汎化率':<15}")
    print("-"*100)

    for name, results in results_dict.items():
        same_stats = results['same_period_stats']
        future_stats = results['future_period_stats']
        if 'f1_score' in same_stats and 'f1_score' in future_stats:
            f1_drop = same_stats['f1_score']['mean'] - future_stats['f1_score']['mean']
            prec_drop = same_stats['precision']['mean'] - future_stats['precision']['mean']
            rec_drop = same_stats['recall']['mean'] - future_stats['recall']['mean']
            generalization_rate = (future_stats['f1_score']['mean'] / same_stats['f1_score']['mean']) * 100

            print(f"{name:<40} {f1_drop:>+.4f}{'':>7} {prec_drop:>+.4f}{'':>7} {rec_drop:>+.4f}{'':>7} {generalization_rate:>6.2f}%")

    # 4. 詳細な時系列分析
    print("\n【4. 訓練期間別の性能推移】")
    for name, results in results_dict.items():
        print(f"\n{name}:")
        print("-"*100)
        print(f"{'訓練期間':<12} {'評価期間':<12} {'F1':<10} {'Precision':<10} {'Recall':<10} {'AUC-ROC':<10} {'サンプル数':<10}")
        print("-"*100)

        for eval_result in results['all_evaluations']:
            train = eval_result['train']
            eval_p = eval_result['eval']
            m = eval_result['metrics']

            samples = f"{m['positive_count']}+/{m['negative_count']}-" if 'positive_count' in m else "N/A"

            print(f"{train:<12} {eval_p:<12} {m['f1_score']:<10.4f} {m['precision']:<10.4f} "
                  f"{m['recall']:<10.4f} {m['auc_roc']:<10.4f} {samples:<10}")

def analyze_oversampling_effect(results_dict: Dict[str, Dict]):
    """オーバーサンプリングの効果分析"""
    print("\n" + "="*100)
    print("【オーバーサンプリングの効果分析】")
    print("="*100)

    # 複数プロジェクトの結果のみを抽出
    multiproject_results = {k: v for k, v in results_dict.items() if 'multiproject' in k.lower()}

    if len(multiproject_results) >= 2:
        print("\n1. オーバーサンプリング倍率による性能変化（複数プロジェクト）")
        print("-"*100)

        base_name = None
        os2x_name = None
        os3x_name = None

        for name in multiproject_results.keys():
            if 'os2x' in name:
                os2x_name = name
            elif 'os3x' in name:
                os3x_name = name
            elif '2020_2024' in name:
                base_name = name

        if base_name and (os2x_name or os3x_name):
            base_f1_same = multiproject_results[base_name]['same_period_stats']['f1_score']['mean']
            base_f1_future = multiproject_results[base_name]['future_period_stats']['f1_score']['mean']

            print(f"\nベースライン（オーバーサンプリングなし）:")
            print(f"  同一期間F1: {base_f1_same:.4f}")
            print(f"  未来期間F1: {base_f1_future:.4f}")

            if os2x_name:
                os2x_f1_same = multiproject_results[os2x_name]['same_period_stats']['f1_score']['mean']
                os2x_f1_future = multiproject_results[os2x_name]['future_period_stats']['f1_score']['mean']
                print(f"\n2倍オーバーサンプリング:")
                print(f"  同一期間F1: {os2x_f1_same:.4f} (変化: {(os2x_f1_same - base_f1_same)*100:+.2f}%)")
                print(f"  未来期間F1: {os2x_f1_future:.4f} (変化: {(os2x_f1_future - base_f1_future)*100:+.2f}%)")

            if os3x_name:
                os3x_f1_same = multiproject_results[os3x_name]['same_period_stats']['f1_score']['mean']
                os3x_f1_future = multiproject_results[os3x_name]['future_period_stats']['f1_score']['mean']
                print(f"\n3倍オーバーサンプリング:")
                print(f"  同一期間F1: {os3x_f1_same:.4f} (変化: {(os3x_f1_same - base_f1_same)*100:+.2f}%)")
                print(f"  未来期間F1: {os3x_f1_future:.4f} (変化: {(os3x_f1_future - base_f1_future)*100:+.2f}%)")

def main():
    results_dir = Path("/Users/kazuki-h/research/multiproject_research/results")

    # 各実験の結果を読み込み
    experiments = {
        'Single Project (Nova)': results_dir / "review_continuation_cross_eval_nova",
        'Multi-Project (No Oversampling)': results_dir / "cross_temporal_openstack_multiproject_2020_2024",
        'Multi-Project (2x Oversampling)': results_dir / "cross_temporal_openstack_multiproject_os2x",
        'Multi-Project (3x Oversampling)': results_dir / "cross_temporal_openstack_multiproject_os3x",
    }

    results_dict = {}

    for name, path in experiments.items():
        if path.exists():
            print(f"分析中: {name}")
            results_dict[name] = analyze_cross_temporal_performance(str(path), name)
        else:
            print(f"見つかりません: {path}")

    # 比較表の出力
    print_comparison_table(results_dict)

    # オーバーサンプリング効果の分析
    analyze_oversampling_effect(results_dict)

    # 主要な考察ポイント
    print("\n" + "="*100)
    print("【主要な考察ポイント】")
    print("="*100)

    print("\n1. 単一プロジェクト vs 複数プロジェクト")
    print("   - データ量の違いによる学習効果")
    print("   - ドメイン多様性による汎化性能")
    print("   - クラス不均衡の程度")

    print("\n2. オーバーサンプリングの効果")
    print("   - クラス不均衡への対処効果")
    print("   - 過学習のリスク")
    print("   - 最適なオーバーサンプリング倍率")

    print("\n3. 時系列での性能変化")
    print("   - コンセプトドリフトの影響")
    print("   - 汎化性能の持続性")
    print("   - 訓練期間の長さの影響")

    print("\n" + "="*100)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
IRL-only正解の全16ケースの詳細データを抽出

使い方:
    python extract_all_irl_only_correct.py
"""

import sys
import pandas as pd
from pathlib import Path
from extract_developer_continuation_data import extract_developer_data

def extract_all_irl_only_correct():
    """
    IRL-only正解の全16ケースの詳細データを抽出
    """

    # IRL-only正解リストを読み込み
    irl_only_file = Path('/Users/kazuki-h/research/multiproject_research/outputs/singleproject/irl_rf_10pattern_analysis/irl_only_correct_unique.csv')
    df = pd.read_csv(irl_only_file)

    print('=' * 80)
    print('IRL-only正解 全16ケースの詳細データ抽出')
    print('=' * 80)
    print(f'総ケース数: {len(df)}')
    print()

    # 出力ディレクトリ
    output_dir = Path('/Users/kazuki-h/research/multiproject_research/outputs/singleproject/irl_only_correct_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 各ケースのサマリーを保存
    summaries = []

    for idx, row in df.iterrows():
        reviewer_email = row['reviewer_id']
        train_period = row['train_period']
        eval_period = row['eval_period']
        pattern = row['pattern']
        true_label = int(row['true_label'])
        irl_prob = row['irl_prob']
        rf_prob = row['rf_prob']
        history_count = int(row['history_count'])
        history_rate = row['history_rate']

        print(f'\n【ケース {idx + 1}/{len(df)}】')
        print(f'パターン: {pattern}')
        print(f'開発者: {reviewer_email}')
        print(f'真のラベル: {"継続" if true_label == 1 else "離脱"}')
        print(f'IRL確率: {irl_prob:.3f} → {"継続" if irl_prob >= 0.471 else "離脱"}')
        print(f'RF確率: {rf_prob:.2f} → {"継続" if rf_prob >= 0.5 else "離脱"}')
        print(f'履歴: {history_count}件, 受諾率{history_rate*100:.1f}%')
        print('-' * 80)

        try:
            # データ抽出
            result = extract_developer_data(
                reviewer_email=reviewer_email,
                train_period=train_period,
                eval_period=eval_period
            )

            history_df = result['history_df']
            eval_df = result['eval_df']

            # サマリー情報
            if len(eval_df) > 0:
                eval_responded = eval_df['label'].sum()
            else:
                eval_responded = 0

            summary = {
                'case_id': idx + 1,
                'pattern': pattern,
                'reviewer_email': reviewer_email,
                'true_label': true_label,
                'irl_prob': irl_prob,
                'rf_prob': rf_prob,
                'history_requests': len(history_df),
                'history_responses': int(history_df['label'].sum()) if len(history_df) > 0 else 0,
                'history_rate': history_df['label'].mean() if len(history_df) > 0 else 0,
                'eval_requests': len(eval_df),
                'eval_responses': int(eval_responded),
                'eval_rate': eval_df['label'].mean() if len(eval_df) > 0 else 0,
            }

            # 時系列情報
            if len(history_df) > 0:
                history_df['request_time'] = pd.to_datetime(history_df['request_time'])
                summary['history_first_date'] = history_df['request_time'].min().strftime('%Y-%m-%d')
                summary['history_last_date'] = history_df['request_time'].max().strftime('%Y-%m-%d')
                summary['history_days_span'] = (history_df['request_time'].max() - history_df['request_time'].min()).days
            else:
                summary['history_first_date'] = None
                summary['history_last_date'] = None
                summary['history_days_span'] = 0

            if len(eval_df) > 0:
                eval_df['request_time'] = pd.to_datetime(eval_df['request_time'])
                summary['eval_first_date'] = eval_df['request_time'].min().strftime('%Y-%m-%d')
                summary['eval_last_date'] = eval_df['request_time'].max().strftime('%Y-%m-%d')
            else:
                summary['eval_first_date'] = None
                summary['eval_last_date'] = None

            # レビュアーの過去実績（履歴期間末時点）
            if len(history_df) > 0:
                last_row = history_df.iloc[-1]
                summary['reviewer_past_reviews_30d'] = last_row['reviewer_past_reviews_30d']
                summary['reviewer_past_reviews_90d'] = last_row['reviewer_past_reviews_90d']
                summary['reviewer_past_reviews_180d'] = last_row['reviewer_past_reviews_180d']
                summary['reviewer_past_response_rate_180d'] = last_row['reviewer_past_response_rate_180d']
                summary['reviewer_tenure_days'] = last_row['reviewer_tenure_days']
            else:
                summary['reviewer_past_reviews_30d'] = 0
                summary['reviewer_past_reviews_90d'] = 0
                summary['reviewer_past_reviews_180d'] = 0
                summary['reviewer_past_response_rate_180d'] = 0
                summary['reviewer_tenure_days'] = 0

            summaries.append(summary)

            print(f'✓ 抽出完了')

        except Exception as e:
            print(f'✗ エラー: {e}')
            continue

    # サマリーをCSV保存
    summary_df = pd.DataFrame(summaries)
    summary_file = output_dir / 'irl_only_correct_detailed_summary.csv'
    summary_df.to_csv(summary_file, index=False)

    print('\n' + '=' * 80)
    print(f'✅ 全{len(summaries)}ケースの抽出完了')
    print(f'サマリー保存: {summary_file}')
    print('=' * 80)

    # 統計情報
    print('\n【全体統計】')
    print(f'\n1. 真のラベル分布:')
    print(f'   継続 (1): {summary_df["true_label"].sum()}件 ({summary_df["true_label"].sum()/len(summary_df)*100:.1f}%)')
    print(f'   離脱 (0): {(summary_df["true_label"] == 0).sum()}件 ({(summary_df["true_label"] == 0).sum()/len(summary_df)*100:.1f}%)')

    print(f'\n2. 履歴期間の統計:')
    print(f'   平均リクエスト数: {summary_df["history_requests"].mean():.1f}件')
    print(f'   平均応答数: {summary_df["history_responses"].mean():.1f}件')
    print(f'   平均受諾率: {summary_df["history_rate"].mean()*100:.1f}%')
    print(f'   平均期間: {summary_df["history_days_span"].mean():.0f}日')

    print(f'\n3. 評価期間の統計:')
    print(f'   平均リクエスト数: {summary_df["eval_requests"].mean():.1f}件')
    print(f'   平均応答数: {summary_df["eval_responses"].mean():.1f}件')

    # 継続者と離脱者で分けて統計
    continuers = summary_df[summary_df['true_label'] == 1]
    churners = summary_df[summary_df['true_label'] == 0]

    print(f'\n4. 継続者 (label=1) の特徴:')
    print(f'   件数: {len(continuers)}')
    print(f'   平均履歴リクエスト: {continuers["history_requests"].mean():.1f}件')
    print(f'   平均履歴受諾率: {continuers["history_rate"].mean()*100:.1f}%')
    print(f'   平均IRL確率: {continuers["irl_prob"].mean():.3f}')
    print(f'   平均RF確率: {continuers["rf_prob"].mean():.2f}')

    print(f'\n5. 離脱者 (label=0) の特徴:')
    print(f'   件数: {len(churners)}')
    print(f'   平均履歴リクエスト: {churners["history_requests"].mean():.1f}件')
    print(f'   平均履歴受諾率: {churners["history_rate"].mean()*100:.1f}%')
    print(f'   平均IRL確率: {churners["irl_prob"].mean():.3f}')
    print(f'   平均RF確率: {churners["rf_prob"].mean():.2f}')

    return summary_df


if __name__ == '__main__':
    summary_df = extract_all_irl_only_correct()

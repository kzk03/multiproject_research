#!/usr/bin/env python3
"""
継続判定に使われた特徴量を詳しく表示するスクリプト

使い方:
    python show_continuation_features.py <開発者メール> <訓練期間> <評価期間>
"""

import sys
import pandas as pd
from pathlib import Path

def show_continuation_features(reviewer_email, train_period, eval_period):
    """
    継続判定に使われた特徴量を詳しく表示
    """

    # 抽出済みデータを読み込み
    data_dir = Path('/Users/kazuki-h/research/multiproject_research/outputs/singleproject/developer_data')
    safe_email = reviewer_email.replace('@', '_at_').replace('.', '_')

    history_file = data_dir / f'{safe_email}_{train_period}_to_{eval_period}_history.csv'
    eval_file = data_dir / f'{safe_email}_{train_period}_to_{eval_period}_eval.csv'

    if not history_file.exists():
        print(f'エラー: {history_file} が見つかりません')
        print(f'先に extract_developer_continuation_data.py を実行してください')
        return

    history_df = pd.read_csv(history_file)
    eval_df = pd.read_csv(eval_file)

    print('=' * 80)
    print(f'継続判定特徴量の詳細: {reviewer_email}')
    print('=' * 80)
    print(f'パターン: {train_period} → {eval_period}')
    print()

    # 履歴期間の統計
    print('【履歴期間の特徴量】')
    print()
    print(f'1. リクエスト数: {len(history_df)} 件')
    print()

    # label=1の統計
    responded = history_df['label'].sum()
    print(f'2. 応答数（承諾数）:')
    print(f'   - 応答回数: {int(responded)} 回')
    print(f'   - 応答率（受諾率）: {responded/len(history_df)*100:.1f}%')
    print()

    # 時間的統計
    history_df['request_time'] = pd.to_datetime(history_df['request_time'])
    if len(history_df) > 0:
        first_req = history_df['request_time'].min()
        last_req = history_df['request_time'].max()
        days_span = (last_req - first_req).days

        print(f'3. 時間的パターン:')
        print(f'   - 最初のリクエスト: {first_req.strftime("%Y-%m-%d")}')
        print(f'   - 最後のリクエスト: {last_req.strftime("%Y-%m-%d")}')
        print(f'   - 期間: {days_span} 日間')

        # 月別分布
        history_df['month'] = history_df['request_time'].dt.to_period('M')
        monthly = history_df.groupby('month').agg({
            'label': ['count', 'sum']
        })
        print(f'\n   月別分布:')
        for month, row in monthly.iterrows():
            count = int(row['label']['count'])
            responded_month = int(row['label']['sum'])
            print(f'     {month}: {count}件 ({responded_month}件応答)')
    print()

    # レビュアーの過去実績（最後の行から）
    if len(history_df) > 0:
        last_row = history_df.iloc[-1]
        print(f'4. レビュアーの過去実績（履歴期間末時点）:')
        print(f'   - reviewer_past_reviews_30d: {last_row["reviewer_past_reviews_30d"]}')
        print(f'   - reviewer_past_reviews_90d: {last_row["reviewer_past_reviews_90d"]}')
        print(f'   - reviewer_past_reviews_180d: {last_row["reviewer_past_reviews_180d"]}')
        print(f'   - reviewer_past_response_rate_180d: {last_row["reviewer_past_response_rate_180d"]:.3f}')
        print(f'   - reviewer_tenure_days: {last_row["reviewer_tenure_days"]:.0f} 日')
    print()

    # 評価期間の統計
    print('=' * 80)
    print('【評価期間の活動】')
    print()
    eval_df['request_time'] = pd.to_datetime(eval_df['request_time'])
    eval_responded = eval_df['label'].sum()

    print(f'1. リクエスト数: {len(eval_df)} 件')
    print(f'2. 応答数: {int(eval_responded)} 回')
    print(f'3. 応答率: {eval_responded/len(eval_df)*100:.1f}%')
    print()

    # 継続ラベル
    continuation_label = 1 if eval_responded > 0 else 0
    label_str = '継続 (1)' if continuation_label == 1 else '離脱 (0)'
    print(f'【継続判定ラベル】')
    print(f'  → {label_str}')
    print()
    print(f'判定基準:')
    print(f'  評価期間に1回でも応答（label=1）があれば → 継続 (1)')
    print(f'  評価期間に応答がなければ → 離脱 (0)')
    print()

    if eval_responded > 0:
        print(f'応答した{int(eval_responded)}件の詳細:')
        responded_rows = eval_df[eval_df['label'] == 1]
        for idx, row in responded_rows.iterrows():
            print(f'  - {row["request_time"].strftime("%Y-%m-%d %H:%M")}')
            print(f'    from: {row["owner_email"]}')
            print(f'    Change ID: {row["change_id"]}')
            if pd.notna(row['first_response_time']):
                print(f'    応答時刻: {row["first_response_time"]}')
            print()

    # IRL/RFが使う特徴量の推測
    print('=' * 80)
    print('【モデルが使う特徴量（推測）】')
    print('=' * 80)
    print()
    print('IRL (LSTM) が使う可能性のある特徴:')
    print(f'  1. リクエスト数: {len(history_df)}')
    print(f'  2. 受諾率: {responded/len(history_df):.3f} ({responded/len(history_df)*100:.1f}%)')
    print(f'  3. 時系列パターン: {days_span}日間の活動')
    print(f'  4. 最終活動からの経過: 評価開始までの日数')
    print(f'  5. 過去のレビュー実績（30d/90d/180d）')
    print(f'  6. 応答率の推移（時系列）')
    print()

    print('RF が使う可能性のある特徴:')
    print(f'  1. リクエスト数: {len(history_df)}')
    print(f'  2. 受諾率: {responded/len(history_df):.3f}')
    print(f'  3. 過去レビュー数（集計値）')
    print(f'  4. 在籍期間')
    print(f'  （時系列パターンは考慮しない）')
    print()

    print('✅ 完了')


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('使い方: python show_continuation_features.py <開発者メール> <訓練期間> <評価期間>')
        print()
        print('例:')
        print('  python show_continuation_features.py christian.rohmann@inovex.de 9-12m 9-12m')
        sys.exit(1)

    reviewer_email = sys.argv[1]
    train_period = sys.argv[2]
    eval_period = sys.argv[3]

    show_continuation_features(reviewer_email, train_period, eval_period)

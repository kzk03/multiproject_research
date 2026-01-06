#!/usr/bin/env python3
"""
開発者の継続判定に使われた具体的な行を抽出するスクリプト

使い方:
    python extract_developer_continuation_data.py <開発者メールアドレス> <訓練期間> <評価期間>

例:
    python extract_developer_continuation_data.py christian.rohmann@inovex.de 9-12m 9-12m
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

def get_period_dates(period_str, base_year=2023):
    """
    期間文字列から開始日と終了日を取得

    Args:
        period_str: '0-3m', '3-6m', '6-9m', '9-12m'
        base_year: 基準年（評価期間用）

    Returns:
        (start_date, end_date) のタプル
    """
    period_map = {
        '0-3m': (f'{base_year}-01-01', f'{base_year}-04-01'),
        '3-6m': (f'{base_year}-04-01', f'{base_year}-07-01'),
        '6-9m': (f'{base_year}-07-01', f'{base_year}-10-01'),
        '9-12m': (f'{base_year}-10-01', f'{base_year + 1}-01-01'),
    }
    return period_map.get(period_str)


def extract_developer_data(reviewer_email, train_period, eval_period,
                           data_path='/Users/kazuki-h/research/multiproject_research/data/review_requests_openstack_multi_5y_detail.csv',
                           output_dir='/Users/kazuki-h/research/multiproject_research/outputs/singleproject/developer_data'):
    """
    開発者の継続判定に使われたデータを抽出

    Args:
        reviewer_email: 開発者のメールアドレス
        train_period: 訓練期間 ('9-12m' など)
        eval_period: 評価期間 ('9-12m' など)
        data_path: データファイルのパス
        output_dir: 出力ディレクトリ
    """

    # データ読み込み
    print(f'データ読み込み: {data_path}')
    df = pd.read_csv(data_path)
    df['request_time'] = pd.to_datetime(df['request_time'])

    # Novaプロジェクトのみフィルタ
    df = df[df['project'] == 'openstack/nova'].copy()

    # 開発者のデータをフィルタ
    dev_df = df[df['reviewer_email'] == reviewer_email].copy()
    dev_df = dev_df.sort_values('request_time')

    print(f'\n開発者: {reviewer_email}')
    print(f'総リクエスト数: {len(dev_df)} 件')
    print()

    # 訓練期間の日付（訓練データは2021年）
    train_start, train_end = get_period_dates(train_period, base_year=2021)

    # 評価期間の日付（評価データは2023年）
    eval_start, eval_end = get_period_dates(eval_period, base_year=2023)

    # 履歴期間: 評価開始日の12ヶ月前から評価開始日まで
    eval_start_dt = pd.to_datetime(eval_start)
    history_start = eval_start_dt - pd.DateOffset(months=12)
    history_end = eval_start_dt

    print('=' * 80)
    print(f'継続判定パターン: {train_period} → {eval_period}')
    print('=' * 80)
    print()
    print(f'訓練期間: {train_start} ～ {train_end}')
    print(f'評価期間: {eval_start} ～ {eval_end}')
    print(f'履歴期間: {history_start.strftime("%Y-%m-%d")} ～ {history_end.strftime("%Y-%m-%d")}')
    print()

    # 履歴期間のデータ
    history_df = dev_df[(dev_df['request_time'] >= history_start) &
                        (dev_df['request_time'] < history_end)]

    # 評価期間のデータ
    eval_df = dev_df[(dev_df['request_time'] >= eval_start) &
                     (dev_df['request_time'] < eval_end)]

    # 統計
    print('【履歴期間のデータ】')
    print(f'  リクエスト数: {len(history_df)} 件')
    if len(history_df) > 0:
        history_responded = history_df['label'].sum()
        print(f'  応答数（label=1）: {int(history_responded)} 件')
        print(f'  応答率: {history_responded/len(history_df)*100:.1f}%')
    print()

    print('【評価期間のデータ】')
    print(f'  リクエスト数: {len(eval_df)} 件')
    if len(eval_df) > 0:
        eval_responded = eval_df['label'].sum()
        print(f'  応答数（label=1）: {int(eval_responded)} 件')
        print(f'  応答率: {eval_responded/len(eval_df)*100:.1f}%')

        # 継続ラベルの判定
        continuation_label = 1 if eval_responded > 0 else 0
        label_str = '継続 (1)' if continuation_label == 1 else '離脱 (0)'
        print(f'\n  継続判定ラベル: {label_str}')
        if continuation_label == 1:
            print(f'    理由: 評価期間に{int(eval_responded)}回応答している')
        else:
            print(f'    理由: 評価期間に応答がない')
    print()

    # 出力ディレクトリ作成
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ファイル名
    safe_email = reviewer_email.replace('@', '_at_').replace('.', '_')
    history_file = output_path / f'{safe_email}_{train_period}_to_{eval_period}_history.csv'
    eval_file = output_path / f'{safe_email}_{train_period}_to_{eval_period}_eval.csv'

    # 履歴期間のデータを保存
    if len(history_df) > 0:
        history_df.to_csv(history_file, index=False)
        print(f'✓ 履歴期間データ保存: {history_file}')

    # 評価期間のデータを保存
    if len(eval_df) > 0:
        eval_df.to_csv(eval_file, index=False)
        print(f'✓ 評価期間データ保存: {eval_file}')

    # 詳細表示
    if len(history_df) > 0:
        print('\n' + '=' * 80)
        print('履歴期間の詳細（最初の10件）')
        print('=' * 80)
        for idx, row in history_df.head(10).iterrows():
            label_str = '✓ 応答' if row['label'] == 1 else '✗ 無応答'
            owner = row['owner_email'] if pd.notna(row['owner_email']) else 'N/A'
            print(f'{row["request_time"].strftime("%Y-%m-%d %H:%M")} | {label_str} | from: {owner[:40]}')
        if len(history_df) > 10:
            print(f'... 他 {len(history_df) - 10} 件')

    if len(eval_df) > 0:
        print('\n' + '=' * 80)
        print('評価期間の詳細（全件）')
        print('=' * 80)
        for idx, row in eval_df.iterrows():
            label_str = '✓ 応答' if row['label'] == 1 else '✗ 無応答'
            owner = row['owner_email'] if pd.notna(row['owner_email']) else 'N/A'
            print(f'{row["request_time"].strftime("%Y-%m-%d %H:%M")} | {label_str} | from: {owner[:40]}')

    print('\n✅ 完了')

    return {
        'history_df': history_df,
        'eval_df': eval_df,
        'history_file': history_file,
        'eval_file': eval_file,
    }


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('使い方: python extract_developer_continuation_data.py <開発者メール> <訓練期間> <評価期間>')
        print()
        print('例:')
        print('  python extract_developer_continuation_data.py christian.rohmann@inovex.de 9-12m 9-12m')
        print('  python extract_developer_continuation_data.py gibizer@gmail.com 0-3m 6-9m')
        print()
        print('期間:')
        print('  0-3m: 1-4月')
        print('  3-6m: 4-7月')
        print('  6-9m: 7-10月')
        print('  9-12m: 10-1月')
        sys.exit(1)

    reviewer_email = sys.argv[1]
    train_period = sys.argv[2]
    eval_period = sys.argv[3]

    extract_developer_data(reviewer_email, train_period, eval_period)

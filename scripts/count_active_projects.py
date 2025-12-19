#!/usr/bin/env python3
"""
各プラットフォームで2021-2024年にアクティブなプロジェクト数を確認
"""

import requests
import json
import time
from typing import Set

def get_active_projects(gerrit_url: str, prefix: str, start_date: str = "2021-01-01", end_date: str = "2024-01-01") -> Set[str]:
    """指定期間にアクティブなプロジェクトを取得"""
    print(f"\n{'='*80}")
    print(f"サーバー: {gerrit_url}")
    print(f"プレフィックス: {prefix}")
    print(f"期間: {start_date} - {end_date}")
    print('='*80)

    session = requests.Session()
    active_projects = set()

    # 各プラットフォームのプロジェクトリストを取得
    try:
        # プロジェクト一覧を取得
        projects_url = f"{gerrit_url}/projects/"
        response = session.get(projects_url, params={'n': 1000}, timeout=60)

        if response.status_code != 200:
            print(f"エラー: プロジェクト一覧の取得に失敗 (status={response.status_code})")
            return active_projects

        # XSSI protection を除去
        content = response.text
        if content.startswith(")]}'\n"):
            content = content[5:]

        projects = json.loads(content)

        # プレフィックスでフィルタリング
        matching_projects = [p for p in projects.keys() if p.startswith(prefix)]
        print(f"\n{prefix}で始まるプロジェクト数: {len(matching_projects)}")

        # 各プロジェクトで期間中のレビューをチェック
        print(f"\n各プロジェクトの活動状況をチェック中...")
        for i, project in enumerate(matching_projects[:100], 1):  # 最大100件をチェック
            try:
                # 期間中のchangesをクエリ
                query = f"project:{project}+after:{start_date}+before:{end_date}"
                changes_url = f"{gerrit_url}/changes/"

                resp = session.get(
                    changes_url,
                    params={'q': query, 'n': 1},  # 1件だけ取得して存在確認
                    timeout=30
                )

                if resp.status_code == 200:
                    content = resp.text
                    if content.startswith(")]}'\n"):
                        content = content[5:]

                    changes = json.loads(content)

                    if len(changes) > 0:
                        active_projects.add(project)
                        print(f"  [{i}/{len(matching_projects[:100])}] ✓ {project} - アクティブ")
                    else:
                        print(f"  [{i}/{len(matching_projects[:100])}] ✗ {project} - 活動なし")
                else:
                    print(f"  [{i}/{len(matching_projects[:100])}] ? {project} - エラー (status={resp.status_code})")

                # レート制限対策
                time.sleep(0.2)

            except Exception as e:
                print(f"  [{i}/{len(matching_projects[:100])}] ! {project} - 例外: {str(e)[:50]}")
                continue

    except Exception as e:
        print(f"エラー: {str(e)}")

    return active_projects


def main():
    platforms = {
        'Chromium': {
            'url': 'https://chromium-review.googlesource.com',
            'prefix': 'chromium/'
        },
        'Android': {
            'url': 'https://android-review.googlesource.com',
            'prefix': 'platform/'
        },
        'Qt': {
            'url': 'https://codereview.qt-project.org',
            'prefix': 'qt/'
        },
        'OpenStack': {
            'url': 'https://review.opendev.org',
            'prefix': 'openstack/'
        }
    }

    print("="*80)
    print("2021-2024年にアクティブなプロジェクト数の調査")
    print("="*80)

    results = {}

    for platform_name, config in platforms.items():
        active = get_active_projects(
            config['url'],
            config['prefix'],
            "2021-01-01",
            "2024-01-01"
        )

        results[platform_name] = {
            'active_count': len(active),
            'projects': sorted(active)
        }

        print(f"\n{platform_name}: {len(active)}個のアクティブプロジェクト")

    # サマリー
    print("\n" + "="*80)
    print("サマリー")
    print("="*80)

    for platform_name, data in results.items():
        count = data['active_count']
        status = "✓ 50個以上" if count >= 50 else f"⚠️ {count}個のみ（50個未満）"
        print(f"{platform_name:12}: {count:3}個のアクティブプロジェクト - {status}")

    # 詳細を保存
    with open('outputs/platform_comparison/active_projects_count.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n詳細結果: outputs/platform_comparison/active_projects_count.json")


if __name__ == '__main__':
    main()

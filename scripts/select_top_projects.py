#!/usr/bin/env python3
"""
Select top 50 projects from each platform (Chromium, Android, Qt)
Based on project naming patterns and importance
"""

import json
import sys
from pathlib import Path


def load_projects(json_path):
    """Load projects from Gerrit JSON response"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def filter_chromium_projects(projects):
    """Select important Chromium projects - chromium/ prefix ONLY"""
    # Only select projects that start with 'chromium/'
    project_names = [p for p in projects.keys()
                     if p.startswith('chromium/')
                     and projects[p].get('state') == 'ACTIVE']

    # Priority patterns for Chromium (within chromium/ namespace)
    priority_patterns = [
        'chromium/src',
        'chromium/tools',
        'chromium/blink',
        'chromium/deps',
        'chromiumos/',
    ]

    selected = []

    # First add priority projects
    for pattern in priority_patterns:
        matches = [p for p in project_names if pattern in p.lower() and p not in selected]
        selected.extend(matches[:10])  # Top 10 per pattern
        if len(selected) >= 50:
            break

    # Fill remaining with other chromium/ projects
    remaining = [p for p in project_names if p not in selected]
    selected.extend(remaining[:50 - len(selected)])

    return selected[:50]


def filter_android_projects(projects):
    """Select important Android projects - platform/ prefix ONLY"""
    # Only select projects that start with 'platform/'
    project_names = [p for p in projects.keys()
                     if p.startswith('platform/')
                     and projects[p].get('state') == 'ACTIVE']

    priority_patterns = [
        'platform/frameworks/',
        'platform/packages/apps/',
        'platform/packages/modules/',
        'platform/system/',
        'platform/build',
        'platform/art',
        'platform/bionic',
        'platform/development',
        'platform/external/',
        'platform/hardware/',
        'platform/libcore',
        'platform/cts',
        'platform/sdk',
        'platform/tools/',
    ]

    selected = []

    # First add priority projects
    for pattern in priority_patterns:
        matches = [p for p in project_names if pattern in p.lower() and p not in selected]
        selected.extend(matches[:5])  # Top 5 per pattern
        if len(selected) >= 50:
            break

    # Fill remaining with other platform/ projects
    remaining = [p for p in project_names if p not in selected]
    selected.extend(remaining[:50 - len(selected)])

    return selected[:50]


def filter_qt_projects(projects):
    """Select important Qt projects - qt/ prefix ONLY"""
    # Only select projects that start with 'qt/'
    project_names = [p for p in projects.keys()
                     if p.startswith('qt/')
                     and projects[p].get('state') == 'ACTIVE'
                     and 'webkit-snapshots' not in p.lower()]

    priority_patterns = [
        'qt/qtbase',
        'qt/qtdeclarative',
        'qt/qtquick',
        'qt/qtwebengine',
        'qt/qtmultimedia',
        'qt/qttools',
        'qt/qtcreator',
        'qt/qt3d',
        'qt/qtwebsockets',
        'qt/qtserialport',
        'qt/qtscript',
        'qt/qtsvg',
        'qt/qtlocation',
        'qt/qtconnectivity',
        'qt/qtsensors',
        'qt/qtwayland',
        'qt/qtcharts',
        'qt/qtdatavis3d',
        'qt/qtnetworkauth',
        'qt/qtxmlpatterns'
    ]

    selected = []

    # First add priority projects
    for pattern in priority_patterns:
        matches = [p for p in project_names if pattern in p.lower() and p not in selected]
        selected.extend(matches[:2])  # Top 2 per pattern
        if len(selected) >= 50:
            break

    # Fill remaining with other qt/ projects
    remaining = [p for p in project_names if p not in selected]
    selected.extend(remaining[:50 - len(selected)])

    return selected[:50]


def main():
    # Load all projects
    chromium_projects = load_projects('/tmp/chromium_projects.json')
    android_projects = load_projects('/tmp/android_projects.json')
    qt_projects = load_projects('/tmp/qt_projects.json')

    print(f"Total Chromium projects: {len(chromium_projects)}")
    print(f"Total Android projects: {len(android_projects)}")
    print(f"Total Qt projects: {len(qt_projects)}")
    print()

    # Filter to top 50
    chromium_top50 = filter_chromium_projects(chromium_projects)
    android_top50 = filter_android_projects(android_projects)
    qt_top50 = filter_qt_projects(qt_projects)

    # Save to files
    output_dir = Path('/Users/kazuki-h/research/multiproject_research')

    with open(output_dir / 'projects_chromium_50.txt', 'w') as f:
        for project in chromium_top50:
            f.write(f"{project}\n")

    with open(output_dir / 'projects_android_50.txt', 'w') as f:
        for project in android_top50:
            f.write(f"{project}\n")

    with open(output_dir / 'projects_qt_50.txt', 'w') as f:
        for project in qt_top50:
            f.write(f"{project}\n")

    print(f"✓ Selected {len(chromium_top50)} Chromium projects")
    print(f"✓ Selected {len(android_top50)} Android projects")
    print(f"✓ Selected {len(qt_top50)} Qt projects")
    print()

    # Show first 10 from each
    print("Top 10 Chromium projects:")
    for p in chromium_top50[:10]:
        print(f"  - {p}")
    print()

    print("Top 10 Android projects:")
    for p in android_top50[:10]:
        print(f"  - {p}")
    print()

    print("Top 10 Qt projects:")
    for p in qt_top50[:10]:
        print(f"  - {p}")
    print()

    print("Project lists saved to:")
    print(f"  - {output_dir / 'projects_chromium_50.txt'}")
    print(f"  - {output_dir / 'projects_android_50.txt'}")
    print(f"  - {output_dir / 'projects_qt_50.txt'}")


if __name__ == '__main__':
    main()

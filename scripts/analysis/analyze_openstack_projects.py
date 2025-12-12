"""
OpenStackプロジェクトの分析
- 全プロジェクト数
- 各プロジェクトのデータ量
- レビュアー数
- 承諾率
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# スタイル設定
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def analyze_openstack_projects(csv_path: str):
    """OpenStackプロジェクトを分析"""
    print(f"データを読み込み中: {csv_path}")
    print("="*100)

    # データ読み込み（サンプリングして確認）
    df_sample = pd.read_csv(csv_path, nrows=5)
    print("\nカラム一覧:")
    print(df_sample.columns.tolist())
    print("\nサンプルデータ:")
    print(df_sample.head())

    # 全データ読み込み
    print("\n全データを読み込み中...")
    df = pd.read_csv(csv_path)
    print(f"総レコード数: {len(df):,}")

    # プロジェクト列を特定
    project_col = None
    for col in df.columns:
        if 'project' in col.lower():
            project_col = col
            break

    if project_col is None:
        print("\nWarning: プロジェクト列が見つかりません")
        print("利用可能なカラム:", df.columns.tolist())
        return None

    print(f"\nプロジェクト列: {project_col}")

    # プロジェクト数
    projects = df[project_col].unique()
    print(f"\n総プロジェクト数: {len(projects)}")

    # プロジェクトごとの統計
    print("\n" + "="*100)
    print("プロジェクトごとの統計")
    print("="*100)

    project_stats = df.groupby(project_col).agg({
        df.columns[0]: 'count',  # レコード数
    }).rename(columns={df.columns[0]: 'review_count'})

    # レビュアー列を特定
    reviewer_col = None
    for col in df.columns:
        if 'reviewer' in col.lower() or 'email' in col.lower():
            reviewer_col = col
            break

    if reviewer_col:
        reviewer_counts = df.groupby(project_col)[reviewer_col].nunique()
        project_stats['reviewer_count'] = reviewer_counts

    # 承諾/拒否の列を特定
    label_col = None
    for col in df.columns:
        if 'label' in col.lower() or 'accepted' in col.lower() or 'status' in col.lower():
            label_col = col
            break

    if label_col:
        acceptance_rates = df.groupby(project_col)[label_col].mean()
        project_stats['acceptance_rate'] = acceptance_rates

    # ソート（レビュー数降順）
    project_stats = project_stats.sort_values('review_count', ascending=False)

    print(f"\n{'プロジェクト':<30} {'レビュー数':>12} {'レビュアー数':>12} {'承諾率':>10}")
    print("-"*100)

    for project, row in project_stats.iterrows():
        review_count = f"{int(row['review_count']):,}"
        reviewer_count = f"{int(row.get('reviewer_count', 0)):,}" if 'reviewer_count' in row else 'N/A'
        acceptance_rate = f"{row.get('acceptance_rate', 0):.1%}" if 'acceptance_rate' in row else 'N/A'
        print(f"{project:<30} {review_count:>12} {reviewer_count:>12} {acceptance_rate:>10}")

    # 統計サマリー
    print("\n" + "="*100)
    print("統計サマリー")
    print("="*100)
    print(f"総プロジェクト数: {len(projects)}")
    print(f"総レビュー数: {len(df):,}")
    print(f"平均レビュー数/プロジェクト: {len(df) / len(projects):.1f}")

    if 'reviewer_count' in project_stats.columns:
        print(f"総レビュアー数（ユニーク）: {df[reviewer_col].nunique():,}")
        print(f"平均レビュアー数/プロジェクト: {project_stats['reviewer_count'].mean():.1f}")

    if 'acceptance_rate' in project_stats.columns:
        print(f"全体承諾率: {df[label_col].mean():.1%}")
        print(f"プロジェクト間承諾率（平均）: {project_stats['acceptance_rate'].mean():.1%}")
        print(f"プロジェクト間承諾率（中央値）: {project_stats['acceptance_rate'].median():.1%}")
        print(f"プロジェクト間承諾率（範囲）: {project_stats['acceptance_rate'].min():.1%} - {project_stats['acceptance_rate'].max():.1%}")

    # 時間範囲を確認
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    if date_cols:
        print(f"\n時間範囲:")
        for date_col in date_cols[:3]:  # 最初の3つの日付列
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                print(f"  {date_col}: {df[date_col].min()} ~ {df[date_col].max()}")
            except:
                pass

    return df, project_stats, project_col

def visualize_project_distribution(project_stats, output_dir: Path):
    """プロジェクト分布を可視化"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('OpenStack Projects Analysis', fontsize=16, fontweight='bold')

    # 1. レビュー数の分布
    ax1 = axes[0, 0]
    top_n = 20
    top_projects = project_stats.head(top_n)

    bars = ax1.barh(range(len(top_projects)), top_projects['review_count'])
    ax1.set_yticks(range(len(top_projects)))
    ax1.set_yticklabels(top_projects.index, fontsize=9)
    ax1.set_xlabel('Review Count', fontweight='bold')
    ax1.set_title(f'Top {top_n} Projects by Review Count', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')

    # 値を表示
    for i, (idx, row) in enumerate(top_projects.iterrows()):
        ax1.text(row['review_count'], i, f" {int(row['review_count']):,}",
                va='center', fontsize=8)

    # 2. レビュアー数の分布
    if 'reviewer_count' in project_stats.columns:
        ax2 = axes[0, 1]
        top_projects_reviewers = project_stats.nlargest(top_n, 'reviewer_count')

        bars = ax2.barh(range(len(top_projects_reviewers)), top_projects_reviewers['reviewer_count'])
        ax2.set_yticks(range(len(top_projects_reviewers)))
        ax2.set_yticklabels(top_projects_reviewers.index, fontsize=9)
        ax2.set_xlabel('Reviewer Count', fontweight='bold')
        ax2.set_title(f'Top {top_n} Projects by Reviewer Count', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')

        for i, (idx, row) in enumerate(top_projects_reviewers.iterrows()):
            ax2.text(row['reviewer_count'], i, f" {int(row['reviewer_count']):,}",
                    va='center', fontsize=8)

    # 3. 承諾率の分布
    if 'acceptance_rate' in project_stats.columns:
        ax3 = axes[1, 0]
        ax3.hist(project_stats['acceptance_rate'], bins=20, alpha=0.7, edgecolor='black')
        ax3.axvline(project_stats['acceptance_rate'].mean(), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {project_stats["acceptance_rate"].mean():.1%}')
        ax3.axvline(project_stats['acceptance_rate'].median(), color='blue', linestyle='--',
                   linewidth=2, label=f'Median: {project_stats["acceptance_rate"].median():.1%}')
        ax3.set_xlabel('Acceptance Rate', fontweight='bold')
        ax3.set_ylabel('Number of Projects', fontweight='bold')
        ax3.set_title('Distribution of Acceptance Rates', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # 4. 累積分布
    ax4 = axes[1, 1]
    cumsum = project_stats['review_count'].cumsum()
    cumsum_pct = cumsum / cumsum.iloc[-1] * 100

    ax4.plot(range(len(cumsum_pct)), cumsum_pct, linewidth=2)
    ax4.axhline(80, color='red', linestyle='--', label='80%')
    ax4.axhline(90, color='orange', linestyle='--', label='90%')
    ax4.set_xlabel('Number of Projects (sorted by review count)', fontweight='bold')
    ax4.set_ylabel('Cumulative Percentage of Reviews (%)', fontweight='bold')
    ax4.set_title('Cumulative Distribution of Reviews', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 80%のレビューをカバーするプロジェクト数を表示
    idx_80 = (cumsum_pct >= 80).idxmax()
    projects_for_80 = list(cumsum_pct.index).index(idx_80) + 1
    ax4.text(projects_for_80, 80, f' {projects_for_80} projects\n cover 80%',
            fontsize=10, va='center')

    plt.tight_layout()
    plt.savefig(output_dir / 'openstack_projects_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n保存: {output_dir / 'openstack_projects_analysis.png'}")

def main():
    # データパス
    csv_path = "data/openstack_20proj_2020_2024_feat.csv"
    output_dir = Path("docs/figures/project_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 分析実行
    result = analyze_openstack_projects(csv_path)

    if result:
        df, project_stats, project_col = result

        # 可視化
        print("\n可視化を生成中...")
        visualize_project_distribution(project_stats, output_dir)

        # 結果をCSVに保存
        project_stats.to_csv(output_dir / 'openstack_projects_stats.csv')
        print(f"保存: {output_dir / 'openstack_projects_stats.csv'}")

        # 予測可能性の評価
        print("\n" + "="*100)
        print("予測可能性の評価")
        print("="*100)

        # 十分なデータがあるプロジェクト（例: 100レビュー以上）
        min_reviews = 100
        sufficient_data = project_stats[project_stats['review_count'] >= min_reviews]

        print(f"\n最低{min_reviews}レビュー以上のプロジェクト: {len(sufficient_data)}/{len(project_stats)}")
        print(f"これらのプロジェクトで全レビューの {sufficient_data['review_count'].sum() / project_stats['review_count'].sum() * 100:.1f}% をカバー")

        # 推奨事項
        print("\n【予測可能性の評価】")
        print("-"*100)
        print(f"✓ 全{len(project_stats)}プロジェクトで予測可能")
        print(f"✓ ただし、データが少ないプロジェクトは精度が低い可能性")
        print(f"✓ 推奨: 最低{min_reviews}レビュー以上のプロジェクト（{len(sufficient_data)}プロジェクト）で運用")

        # データ量による層別化
        print("\n【プロジェクトの層別化】")
        print("-"*100)

        bins = [0, 100, 500, 1000, 5000, float('inf')]
        labels = ['Very Small (<100)', 'Small (100-500)', 'Medium (500-1K)', 'Large (1K-5K)', 'Very Large (>5K)']

        project_stats['size_category'] = pd.cut(project_stats['review_count'], bins=bins, labels=labels)
        size_dist = project_stats['size_category'].value_counts().sort_index()

        for category, count in size_dist.items():
            subset = project_stats[project_stats['size_category'] == category]
            total_reviews = subset['review_count'].sum()
            pct = total_reviews / project_stats['review_count'].sum() * 100
            print(f"{category:>20}: {count:>3} プロジェクト ({total_reviews:>8,} レビュー, {pct:>5.1f}%)")

if __name__ == "__main__":
    main()

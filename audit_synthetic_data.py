"""
Comprehensive Synthetic Data Quality Audit for Recommender System
Checks for red flags that would hurt credibility with hiring managers.

This audit focuses on practical issues for mid-level positions (3-5 YOE).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

# Configuration
DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("validation_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Results storage
validation_results = {
    "timestamp": datetime.now().isoformat(),
    "checks": {},
    "red_flags": [],
    "warnings": [],
    "good_points": []
}

def check_data_exists():
    """Check if data files exist."""
    print("=" * 60)
    print("SYNTHETIC DATA QUALITY AUDIT")
    print("=" * 60)
    print("\n1. Checking data files...")

    required_files = ['users.csv', 'likes.csv', 'matches.csv']
    missing_files = []

    for file in required_files:
        filepath = DATA_DIR / file
        if not filepath.exists():
            missing_files.append(file)
            print(f"   ❌ Missing: {file}")
        else:
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"   ✅ Found: {file} ({size_mb:.1f} MB)")

    if missing_files:
        print(f"\n⚠️  Missing files: {missing_files}")
        print("   Run: python data/generate_data.py")
        return False

    return True

def load_data():
    """Load all data files."""
    print("\n2. Loading data...")

    users_df = pd.read_csv(DATA_DIR / 'users.csv')
    likes_df = pd.read_csv(DATA_DIR / 'likes.csv')
    matches_df = pd.read_csv(DATA_DIR / 'matches.csv')

    print(f"   Users: {len(users_df):,}")
    print(f"   Likes: {len(likes_df):,}")
    print(f"   Matches: {len(matches_df):,}")

    return users_df, likes_df, matches_df

def check_sparsity(users_df, likes_df):
    """
    CRITICAL CHECK #1: Matrix Sparsity
    RED FLAG if sparsity < 95%
    GOOD if sparsity 97-99%
    """
    print("\n" + "=" * 60)
    print("CHECK 1: MATRIX SPARSITY (THE BIG ONE)")
    print("=" * 60)

    n_users = len(users_df)
    n_items = n_users  # Dating app: users are both users and items
    n_ratings = len(likes_df)

    sparsity = 1 - (n_ratings / (n_users * n_items))

    print(f"\nMatrix dimensions: {n_users:,} users × {n_items:,} items")
    print(f"Total interactions: {n_ratings:,}")
    print(f"Matrix sparsity: {sparsity*100:.2f}%")

    validation_results["checks"]["sparsity"] = {
        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": n_ratings,
        "sparsity_pct": round(sparsity * 100, 2)
    }

    # Evaluation
    if sparsity < 0.95:
        flag = f"🚨 CRITICAL: Matrix too dense ({sparsity*100:.2f}% sparsity)"
        print(f"\n{flag}")
        print("   Real systems (Netflix, Spotify) are 99%+ sparse")
        print("   This will look unrealistic to hiring managers")
        validation_results["red_flags"].append(flag)
        return "FAIL"
    elif sparsity < 0.97:
        warning = f"⚠️  Acceptable but low ({sparsity*100:.2f}% sparsity)"
        print(f"\n{warning}")
        print("   Aim for 97-99% for production-like data")
        validation_results["warnings"].append(warning)
        return "WARNING"
    else:
        good = f"✅ Excellent sparsity ({sparsity*100:.2f}%)"
        print(f"\n{good}")
        print("   Comparable to real production systems")
        validation_results["good_points"].append(good)
        return "PASS"

def check_popularity_distribution(likes_df):
    """
    CRITICAL CHECK #2: Item Popularity Distribution
    Should follow power law (long tail), not uniform
    """
    print("\n" + "=" * 60)
    print("CHECK 2: ITEM POPULARITY DISTRIBUTION")
    print("=" * 60)

    # Count likes per person being liked
    item_counts = likes_df['liked_user_id'].value_counts()

    print(f"\nAnalyzing popularity of {len(item_counts):,} users...")
    print(f"Most popular user: {item_counts.max()} likes")
    print(f"Median popularity: {item_counts.median():.0f} likes")
    print(f"Average popularity: {item_counts.mean():.1f} likes")

    # Gini coefficient (inequality measure)
    def gini(x):
        sorted_x = np.sort(x)
        n = len(x)
        cumsum = np.cumsum(sorted_x)
        return (2 * np.sum((n - np.arange(1, n+1) + 1) * sorted_x)) / (n * cumsum[-1]) - (n + 1) / n

    gini_coefficient = gini(item_counts.values)
    print(f"Gini coefficient: {gini_coefficient:.3f}")

    # Check 80/20 rule
    top_20_pct = int(len(item_counts) * 0.2)
    top_20_ratings = item_counts.head(top_20_pct).sum()
    pct_from_top_20 = top_20_ratings / len(likes_df) * 100
    print(f"Top 20% of users get: {pct_from_top_20:.1f}% of likes")

    validation_results["checks"]["popularity"] = {
        "gini_coefficient": round(gini_coefficient, 3),
        "top_20_pct_share": round(pct_from_top_20, 1),
        "most_popular": int(item_counts.max()),
        "median": int(item_counts.median())
    }

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(item_counts, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Number of Likes Received')
    axes[0].set_ylabel('Number of Users')
    axes[0].set_title('User Popularity Distribution')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)

    # Power law check
    sorted_counts = sorted(item_counts, reverse=True)
    axes[1].plot(range(len(sorted_counts)), sorted_counts, linewidth=2)
    axes[1].set_xlabel('User Rank')
    axes[1].set_ylabel('Number of Likes')
    axes[1].set_title('Power Law Check (Long Tail)')
    axes[1].set_yscale('log')
    axes[1].set_xscale('log')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'popularity_distribution.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Saved: {OUTPUT_DIR / 'popularity_distribution.png'}")

    # Evaluation
    if gini_coefficient < 0.6:
        flag = f"🚨 CRITICAL: Too uniform (Gini={gini_coefficient:.3f})"
        print(f"\n{flag}")
        print("   Real systems have Gini ~0.8-0.9 (strong inequality)")
        validation_results["red_flags"].append(flag)
        return "FAIL"
    elif gini_coefficient < 0.75:
        warning = f"⚠️  Moderate inequality (Gini={gini_coefficient:.3f})"
        print(f"\n{warning}")
        print("   Aim for Gini ~0.8-0.9 for realistic long-tail")
        validation_results["warnings"].append(warning)
        return "WARNING"
    else:
        good = f"✅ Good power-law distribution (Gini={gini_coefficient:.3f})"
        print(f"\n{good}")
        validation_results["good_points"].append(good)
        return "PASS"

def check_rating_distribution(likes_df):
    """
    CHECK 3: Like Distribution (if ratings exist)
    For binary likes, check if distribution is reasonable
    """
    print("\n" + "=" * 60)
    print("CHECK 3: INTERACTION TYPE DISTRIBUTION")
    print("=" * 60)

    # Check if there are different interaction types
    if 'interaction_type' in likes_df.columns:
        type_dist = likes_df['interaction_type'].value_counts()
        print("\nInteraction types:")
        for itype, count in type_dist.items():
            pct = count / len(likes_df) * 100
            print(f"   {itype}: {count:,} ({pct:.1f}%)")

        validation_results["checks"]["interaction_types"] = type_dist.to_dict()

    # Check if there's a rating column
    if 'rating' in likes_df.columns:
        rating_dist = likes_df['rating'].value_counts().sort_index()
        print("\nRating distribution:")
        for rating, count in rating_dist.items():
            pct = count / len(likes_df) * 100
            print(f"   {rating}: {count:,} ({pct:.1f}%)")

        validation_results["checks"]["rating_distribution"] = rating_dist.to_dict()

        # Plot
        plt.figure(figsize=(8, 5))
        (rating_dist / len(likes_df) * 100).plot(kind='bar', color='steelblue', edgecolor='black')
        plt.xlabel('Rating')
        plt.ylabel('Percentage')
        plt.title('Rating Distribution')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'rating_distribution.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Saved: {OUTPUT_DIR / 'rating_distribution.png'}")
    else:
        print("\nNo rating column found (binary likes only)")
        validation_results["checks"]["rating_distribution"] = "Binary likes only"

    return "PASS"

def check_cold_start(users_df, likes_df):
    """
    CRITICAL CHECK #4: Cold Start Scenarios
    Real systems have many users/items with few interactions
    """
    print("\n" + "=" * 60)
    print("CHECK 4: COLD START SCENARIOS")
    print("=" * 60)

    # User activity (giving likes)
    user_like_counts = likes_df.groupby('user_id').size()
    n_users = len(users_df)

    # Users who gave likes
    users_with_activity = len(user_like_counts)
    users_no_activity = n_users - users_with_activity

    print(f"\nUser Activity:")
    print(f"   Users with interactions: {users_with_activity:,}")
    print(f"   Users with NO interactions: {users_no_activity:,}")

    if users_with_activity > 0:
        cold_start_users = (user_like_counts < 5).sum()
        cold_start_pct = cold_start_users / users_with_activity * 100

        print(f"   Users with <5 likes: {cold_start_users:,} ({cold_start_pct:.1f}%)")
        print(f"   Average likes/user: {user_like_counts.mean():.1f}")
        print(f"   Median likes/user: {user_like_counts.median():.0f}")

    # Item popularity (being liked)
    item_like_counts = likes_df.groupby('liked_user_id').size()
    total_items = len(users_df)

    print(f"\nItem Popularity:")
    print(f"   Users ever liked: {len(item_like_counts):,} / {total_items:,}")

    cold_start_items = (item_like_counts < 10).sum()
    cold_start_items_pct = cold_start_items / total_items * 100

    print(f"   Items with <10 likes: {cold_start_items:,} ({cold_start_items_pct:.1f}%)")
    print(f"   Average likes/item: {item_like_counts.mean():.1f}")
    print(f"   Median likes/item: {item_like_counts.median():.0f}")

    validation_results["checks"]["cold_start"] = {
        "cold_start_users_pct": round(cold_start_pct, 1) if users_with_activity > 0 else None,
        "cold_start_items_pct": round(cold_start_items_pct, 1),
        "avg_likes_per_user": round(user_like_counts.mean(), 1) if users_with_activity > 0 else 0,
        "avg_likes_per_item": round(item_like_counts.mean(), 1)
    }

    # Plot distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if users_with_activity > 0:
        axes[0].hist(user_like_counts, bins=50, edgecolor='black', alpha=0.7, color='coral')
        axes[0].set_xlabel('Likes Given per User')
        axes[0].set_ylabel('Number of Users')
        axes[0].set_title('User Activity Distribution')
        axes[0].set_yscale('log')
        axes[0].axvline(5, color='red', linestyle='--', label='Cold Start (<5)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    axes[1].hist(item_like_counts, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[1].set_xlabel('Likes Received per Item')
    axes[1].set_ylabel('Number of Items')
    axes[1].set_title('Item Popularity Distribution')
    axes[1].set_yscale('log')
    axes[1].axvline(10, color='red', linestyle='--', label='Cold Start (<10)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'cold_start_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Saved: {OUTPUT_DIR / 'cold_start_analysis.png'}")

    # Evaluation
    if users_with_activity > 0 and cold_start_pct < 5:
        flag = "🚨 CRITICAL: Almost no cold-start users"
        print(f"\n{flag}")
        print("   Real systems have 20-30% cold-start users")
        validation_results["red_flags"].append(flag)
        return "FAIL"
    elif users_with_activity > 0 and cold_start_pct < 15:
        warning = f"⚠️  Few cold-start users ({cold_start_pct:.1f}%)"
        print(f"\n{warning}")
        validation_results["warnings"].append(warning)
        return "WARNING"
    else:
        good = f"✅ Realistic cold-start distribution"
        print(f"\n{good}")
        validation_results["good_points"].append(good)
        return "PASS"

def check_user_behavior_variance(users_df, likes_df):
    """
    CHECK 5: User Behavior Variance
    Different users should have different patterns
    """
    print("\n" + "=" * 60)
    print("CHECK 5: USER BEHAVIOR VARIANCE")
    print("=" * 60)

    user_like_counts = likes_df.groupby('user_id').size()

    if len(user_like_counts) == 0:
        print("\nNo user activity to analyze")
        return "SKIP"

    print(f"\nUser activity statistics:")
    print(f"   Mean likes/user: {user_like_counts.mean():.1f}")
    print(f"   Std dev: {user_like_counts.std():.1f}")
    print(f"   Min: {user_like_counts.min()}")
    print(f"   Max: {user_like_counts.max()}")

    # Coefficient of variation
    cv = user_like_counts.std() / user_like_counts.mean()
    print(f"   Coefficient of variation: {cv:.2f}")

    validation_results["checks"]["user_variance"] = {
        "mean": round(user_like_counts.mean(), 1),
        "std": round(user_like_counts.std(), 1),
        "coefficient_of_variation": round(cv, 2)
    }

    # Evaluation
    if cv < 0.3:
        warning = f"⚠️  Users behave too similarly (CV={cv:.2f})"
        print(f"\n{warning}")
        print("   Real users have more diverse behavior")
        validation_results["warnings"].append(warning)
        return "WARNING"
    else:
        good = f"✅ Good user behavior diversity (CV={cv:.2f})"
        print(f"\n{good}")
        validation_results["good_points"].append(good)
        return "PASS"

def check_temporal_patterns(likes_df):
    """
    CHECK 6: Temporal Patterns
    Check if timestamps exist and are realistic
    """
    print("\n" + "=" * 60)
    print("CHECK 6: TEMPORAL PATTERNS")
    print("=" * 60)

    if 'timestamp' not in likes_df.columns:
        print("\nNo timestamp column found")
        validation_results["checks"]["temporal"] = "No timestamps"
        return "SKIP"

    likes_df['timestamp'] = pd.to_datetime(likes_df['timestamp'])

    # Time range
    time_range = (likes_df['timestamp'].max() - likes_df['timestamp'].min()).days
    print(f"\nTime range: {time_range} days")
    print(f"First interaction: {likes_df['timestamp'].min()}")
    print(f"Last interaction: {likes_df['timestamp'].max()}")

    # Daily activity
    daily_counts = likes_df.groupby(likes_df['timestamp'].dt.date).size()
    print(f"\nDaily activity:")
    print(f"   Mean: {daily_counts.mean():.0f} likes/day")
    print(f"   Std dev: {daily_counts.std():.0f}")
    print(f"   CV: {daily_counts.std()/daily_counts.mean():.2f}")

    validation_results["checks"]["temporal"] = {
        "time_range_days": time_range,
        "avg_daily_interactions": round(daily_counts.mean(), 0),
        "daily_std": round(daily_counts.std(), 0)
    }

    good = f"✅ Temporal data present ({time_range} days)"
    print(f"\n{good}")
    validation_results["good_points"].append(good)

    return "PASS"

def generate_validation_report():
    """Generate final validation report."""
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    print(f"\n✅ GOOD POINTS ({len(validation_results['good_points'])}):")
    for point in validation_results['good_points']:
        print(f"   • {point}")

    if validation_results['warnings']:
        print(f"\n⚠️  WARNINGS ({len(validation_results['warnings'])}):")
        for warning in validation_results['warnings']:
            print(f"   • {warning}")

    if validation_results['red_flags']:
        print(f"\n🚨 CRITICAL ISSUES ({len(validation_results['red_flags'])}):")
        for flag in validation_results['red_flags']:
            print(f"   • {flag}")

    # Overall assessment
    print("\n" + "=" * 60)
    if len(validation_results['red_flags']) > 0:
        print("OVERALL ASSESSMENT: ❌ NEEDS FIXES")
        print("\nHiring managers WILL notice these issues.")
        print("Recommendation: Fix critical issues before showcasing.")
    elif len(validation_results['warnings']) > 1:
        print("OVERALL ASSESSMENT: ⚠️  ACCEPTABLE WITH IMPROVEMENTS")
        print("\nGood enough for portfolio, but improvements recommended.")
    else:
        print("OVERALL ASSESSMENT: ✅ PRODUCTION-QUALITY")
        print("\nData quality is credible for mid-level position.")
    print("=" * 60)

    # Save JSON report
    report_file = OUTPUT_DIR / 'validation_report.json'
    with open(report_file, 'w') as f:
        json.dump(validation_results, f, indent=2)
    print(f"\n📄 Detailed report saved: {report_file}")

def main():
    """Run full audit."""
    if not check_data_exists():
        print("\n⚠️  Cannot proceed without data files.")
        print("Generate data first: python data/generate_data.py")
        return

    users_df, likes_df, matches_df = load_data()

    # Run all checks
    check_sparsity(users_df, likes_df)
    check_popularity_distribution(likes_df)
    check_rating_distribution(likes_df)
    check_cold_start(users_df, likes_df)
    check_user_behavior_variance(users_df, likes_df)
    check_temporal_patterns(likes_df)

    # Generate report
    generate_validation_report()

    print(f"\n📁 All results saved to: {OUTPUT_DIR}/")
    print("   • validation_report.json")
    print("   • popularity_distribution.png")
    print("   • cold_start_analysis.png")

if __name__ == "__main__":
    main()

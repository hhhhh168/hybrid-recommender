"""
Diagnostic script to understand what's happening with exclusions.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor

def main():
    print("="*70)
    print("DIAGNOSTIC: Understanding Evaluation Exclusions")
    print("="*70)

    # Load data
    print("\n1. Loading data...")
    loader = DataLoader()
    users_df, likes_df, _ = loader.load_all()
    print(f"   Total users: {len(users_df):,}")
    print(f"   Total likes: {len(likes_df):,}")

    # Train/test split
    print("\n2. Creating train/test split...")
    preprocessor = DataPreprocessor(loader)
    train_likes, test_likes = preprocessor.train_test_split_temporal(likes_df, test_size=0.2)
    print(f"   Train: {len(train_likes):,} likes")
    print(f"   Test: {len(test_likes):,} likes")

    # Build ground truth and training interactions
    print("\n3. Building ground truth and training interactions...")
    ground_truth = {}
    for uid, grp in test_likes.groupby('user_id'):
        ground_truth[uid] = set(grp['liked_user_id'].unique())

    train_interactions = {}
    for uid, grp in train_likes.groupby('user_id'):
        train_interactions[uid] = set(grp['liked_user_id'].unique())

    print(f"   Test users with ground truth: {len(ground_truth):,}")
    print(f"   Users with training interactions: {len(train_interactions):,}")

    # Analyze overlap for sample users
    print("\n4. Analyzing exclusion behavior...")
    print("="*70)

    # Sample 10 random test users
    sample_users = np.random.choice(list(ground_truth.keys()), size=min(10, len(ground_truth)), replace=False)

    total_candidates = len(users_df)

    for user_id in sample_users:
        test_likes_set = ground_truth.get(user_id, set())
        train_likes_set = train_interactions.get(user_id, set())

        # What we're currently excluding
        current_exclusions = train_likes_set | {user_id}

        # What we SHOULD exclude (all historical likes from FULL dataset)
        all_historical_likes = set(likes_df[likes_df['user_id'] == user_id]['liked_user_id'].unique())

        # Remaining candidates
        remaining_with_current = total_candidates - len(current_exclusions)
        remaining_with_correct = total_candidates - len(all_historical_likes) - 1  # -1 for self

        # Overlap between train and test
        overlap = train_likes_set & test_likes_set

        print(f"\nUser: {user_id[:8]}...")
        print(f"  Ground truth (test likes):     {len(test_likes_set):4d}")
        print(f"  Training likes:                {len(train_likes_set):4d}")
        print(f"  Overlap (contamination):       {len(overlap):4d}")
        print(f"  Current exclusions:            {len(current_exclusions):4d}")
        print(f"  Remaining candidates:          {remaining_with_current:4d}")
        print(f"  ---")
        print(f"  ALL historical likes:          {len(all_historical_likes):4d}")
        print(f"  Should have remaining:         {remaining_with_correct:4d}")
        print(f"  ---")
        print(f"  Problem: Excluding ONLY train = test items still in pool!")

    # Summary statistics
    print("\n" + "="*70)
    print("5. SUMMARY STATISTICS")
    print("="*70)

    overlaps = []
    exclusion_ratios = []

    for user_id in ground_truth.keys():
        test_set = ground_truth[user_id]
        train_set = train_interactions.get(user_id, set())
        overlap = len(train_set & test_set)
        overlaps.append(overlap)

        exclusions = len(train_set) + 1  # train + self
        exclusion_ratios.append(exclusions / total_candidates)

    print(f"\nAverage overlap (train ∩ test):     {np.mean(overlaps):.2f} users")
    print(f"Median overlap:                      {np.median(overlaps):.2f} users")
    print(f"Max overlap:                         {np.max(overlaps):.0f} users")
    print(f"\nAverage exclusion ratio:             {np.mean(exclusion_ratios):.1%}")
    print(f"Average remaining candidates:        {total_candidates * (1 - np.mean(exclusion_ratios)):.0f}")

    print("\n" + "="*70)
    print("6. ROOT CAUSE ANALYSIS")
    print("="*70)

    print("\nThe problem is:")
    print("  We're excluding TRAINING interactions")
    print("  ✓ We should exclude ALL HISTORICAL interactions (train + test)")
    print("")
    print("Why this breaks evaluation:")
    print("  • Test set contains NEW likes from the test period")
    print("  • But many of these 'new' likes are to people they liked in training")
    print("  • By excluding training likes, we allow re-recommendation of already-liked users")
    print("  • This inflates metrics artificially")
    print("")
    print("The fix:")
    print("  • Build exclude_ids from BOTH train AND test likes (full history)")
    print("  • This ensures we only recommend truly novel users")
    print("")

    # Check if this is indeed the issue
    all_likes_per_user = likes_df.groupby('user_id')['liked_user_id'].apply(set)

    sample_user = sample_users[0]
    test_gt = ground_truth[sample_user]
    train_excl = train_interactions.get(sample_user, set())
    all_hist = all_likes_per_user.get(sample_user, set())

    in_test_but_not_in_train = test_gt - train_excl

    print(f"\nExample user: {sample_user[:8]}...")
    print(f"  Test ground truth size:              {len(test_gt)}")
    print(f"  Training exclusions:                 {len(train_excl)}")
    print(f"  Items in test but NOT in train:      {len(in_test_but_not_in_train)}")
    print(f"  All historical likes:                {len(all_hist)}")
    print("")
    print(f"  → These {len(in_test_but_not_in_train)} items would be recommended!")
    print(f"  → This creates artificially high precision!")
    print("")

    print("="*70)

if __name__ == '__main__':
    main()

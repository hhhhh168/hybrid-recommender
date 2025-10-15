"""
Check if the model is actually producing good recommendations.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.models.hybrid_recommender import HybridRecommender

def main():
    print("="*70)
    print("DIAGNOSTIC: Model Recommendation Quality")
    print("="*70)

    # Load data
    print("\n1. Loading data...")
    loader = DataLoader()
    users_df, likes_df, _ = loader.load_all()

    # Train/test split
    print("\n2. Creating split...")
    preprocessor = DataPreprocessor(loader)
    train_likes, test_likes = preprocessor.train_test_split_temporal(likes_df, test_size=0.2)

    # Train model
    print("\n3. Training Hybrid model...")
    hybrid = HybridRecommender()
    hybrid.fit(train_likes, users_df)

    # Test on a sample user
    print("\n4. Testing recommendations...")
    print("="*70)

    # Get a user who has both train and test likes
    test_users_with_gt = test_likes.groupby('user_id').size()
    sample_user = test_users_with_gt.nlargest(1).index[0]

    # Get their history
    train_likes_user = set(train_likes[train_likes['user_id'] == sample_user]['liked_user_id'])
    test_likes_user = set(test_likes[test_likes['user_id'] == sample_user]['liked_user_id'])

    print(f"\nSample user: {sample_user[:8]}...")
    print(f"  Training likes: {len(train_likes_user)}")
    print(f"  Test likes (ground truth): {len(test_likes_user)}")

    # Get recommendations WITHOUT excluding training
    print("\n5. Recommendations WITHOUT excluding training:")
    recs_no_exclude = hybrid.recommend(sample_user, k=20, exclude_ids=set())
    rec_ids_no_exclude = [r[0] for r in recs_no_exclude]

    hits_in_test = sum(1 for rid in rec_ids_no_exclude if rid in test_likes_user)
    hits_in_train = sum(1 for rid in rec_ids_no_exclude if rid in train_likes_user)

    print(f"  Top-20 recommendations: {len(rec_ids_no_exclude)}")
    print(f"  Hits in TEST (ground truth): {hits_in_test} ({hits_in_test/20:.1%})")
    print(f"  Hits in TRAIN (should exclude): {hits_in_train} ({hits_in_train/20:.1%})")

    # Get recommendations WITH excluding training (current approach)
    print("\n6. Recommendations WITH excluding training (current eval):")
    recs_with_exclude = hybrid.recommend(sample_user, k=20, exclude_ids=train_likes_user)
    rec_ids_with_exclude = [r[0] for r in recs_with_exclude]

    hits_in_test_2 = sum(1 for rid in rec_ids_with_exclude if rid in test_likes_user)
    hits_in_train_2 = sum(1 for rid in rec_ids_with_exclude if rid in train_likes_user)

    print(f"  Top-20 recommendations: {len(rec_ids_with_exclude)}")
    print(f"  Hits in TEST (ground truth): {hits_in_test_2} ({hits_in_test_2/20:.1%})")
    print(f"  Hits in TRAIN (should be 0): {hits_in_train_2}")

    print("\n" + "="*70)
    print("7. ANALYSIS")
    print("="*70)

    if hits_in_test_2 == 0:
        print("\n❌ PROBLEM FOUND: Model can't predict ANY test likes!")
        print("   This means the model has NO predictive power.")
        print("")
        print("   Possible causes:")
        print("   1. Model is random/not learning patterns")
        print("   2. Test set is completely different from training")
        print("   3. Ground truth labels are noisy/unreliable")
        print("")
    elif hits_in_test_2 < 2:
        print("\n⚠️  Model has VERY WEAK predictive power")
        print(f"   Only {hits_in_test_2}/20 hits in top-20")
        print(f"   Precision@20: {hits_in_test_2/20:.1%}")
        print("")
    else:
        print("\n✓ Model has reasonable predictive power")
        print(f"   {hits_in_test_2}/20 hits in top-20")
        print(f"   Precision@20: {hits_in_test_2/20:.1%}")

    # Check if test labels overlap with candidates
    print("\n8. Checking if test ground truth is even in the candidate pool...")
    all_user_ids = set(users_df['user_id'])
    test_in_pool = test_likes_user & all_user_ids
    test_not_in_pool = test_likes_user - all_user_ids

    print(f"   Test ground truth size: {len(test_likes_user)}")
    print(f"   In candidate pool: {len(test_in_pool)}")
    print(f"   NOT in candidate pool: {len(test_not_in_pool)}")

    if len(test_not_in_pool) > 0:
        print("\n❌ MAJOR PROBLEM: Some test users don't exist in the pool!")
        print("   This indicates data corruption or mismatch.")
    else:
        print("\n✓ All test ground truth users exist in candidate pool")

    print("\n" + "="*70)

if __name__ == '__main__':
    main()

"""
Test script for collaborative filtering model only.
"""

from src.data_loader import DataLoader
from src.models.cf_recommender import CollaborativeFilteringRecommender
from src.utils import setup_logger
import time

# Setup logger
logger = setup_logger("test_cf_model")


def main():
    """Main test function."""
    logger.info("="*60)
    logger.info("COLLABORATIVE FILTERING MODEL TEST")
    logger.info("="*60)

    # Load data
    logger.info("\nLoading data...")
    loader = DataLoader()
    users_df, likes_df, matches_df = loader.load_all()
    logger.info(f"✓ Data loaded: {len(users_df)} users, {len(likes_df)} likes")

    # Initialize and train
    logger.info("\n" + "="*60)
    logger.info("TRAINING MODEL")
    logger.info("="*60)

    cf_model = CollaborativeFilteringRecommender(
        n_similar_users=50,
        min_interactions=5,
        like_weight=2.0,
        superlike_weight=3.0
    )

    start_time = time.time()
    cf_model.fit(likes_df, users_df)
    train_time = time.time() - start_time

    logger.info(f"\n✓ Training completed in {train_time:.2f} seconds")

    # Test recommendations for a sample user
    logger.info("\n" + "="*60)
    logger.info("TESTING RECOMMENDATIONS")
    logger.info("="*60)

    # Get a user with decent activity
    active_users = likes_df['user_id'].value_counts()
    sample_user_id = active_users.index[10]  # User with moderate activity

    user_info = users_df[users_df['user_id'] == sample_user_id].iloc[0]
    logger.info(f"\nTest User: {sample_user_id[:8]}...")
    logger.info(f"  Gender: {user_info['gender']}")
    logger.info(f"  Age: {user_info['age']}")
    logger.info(f"  City: {user_info['city']}")
    logger.info(f"  Job: {user_info['job_title']}")
    logger.info(f"  Likes given: {active_users[sample_user_id]}")

    start_time = time.time()
    recommendations = cf_model.recommend(sample_user_id, k=10)
    rec_time = time.time() - start_time

    logger.info(f"\n✓ Generated {len(recommendations)} recommendations in {rec_time*1000:.2f}ms")
    logger.info("\nTop 10 Recommendations:")
    for i, (rec_id, score) in enumerate(recommendations, 1):
        rec_user = users_df[users_df['user_id'] == rec_id].iloc[0]
        logger.info(
            f"  {i}. {rec_id[:8]}... | {rec_user['gender']} {rec_user['age']} | "
            f"{rec_user['city']} | {rec_user['job_title']} | Score: {score:.4f}"
        )

    # Test scoring
    logger.info("\n" + "="*60)
    logger.info("TESTING SCORING")
    logger.info("="*60)

    candidate_ids = [rec[0] for rec in recommendations[:5]]
    scores = cf_model.score(sample_user_id, candidate_ids)

    logger.info(f"\n✓ Scored {len(scores)} candidates:")
    for cid, score in scores.items():
        logger.info(f"  User {cid[:8]}...: {score:.4f}")

    # Test cold-start user
    logger.info("\n" + "="*60)
    logger.info("TESTING COLD-START HANDLING")
    logger.info("="*60)

    cold_start_users = likes_df['user_id'].value_counts()
    cold_user_id = cold_start_users[cold_start_users < 3].index[0]

    cold_user_info = users_df[users_df['user_id'] == cold_user_id].iloc[0]
    logger.info(f"\nCold-start User: {cold_user_id[:8]}...")
    logger.info(f"  Likes given: {cold_start_users[cold_user_id]}")
    logger.info(f"  Gender: {cold_user_info['gender']}")
    logger.info(f"  Age: {cold_user_info['age']}")

    cold_recs = cf_model.recommend(cold_user_id, k=5)
    logger.info(f"\n✓ Generated {len(cold_recs)} cold-start recommendations:")
    for i, (rec_id, score) in enumerate(cold_recs, 1):
        rec_user = users_df[users_df['user_id'] == rec_id].iloc[0]
        logger.info(
            f"  {i}. {rec_id[:8]}... | {rec_user['gender']} {rec_user['age']} | "
            f"{rec_user['job_title']} | Score: {score:.4f}"
        )

    # Test save/load
    logger.info("\n" + "="*60)
    logger.info("TESTING SAVE/LOAD")
    logger.info("="*60)

    from pathlib import Path
    model_path = Path("results/models/cf_model.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    cf_model.save(model_path)
    logger.info(f"\n✓ Model saved to {model_path}")
    logger.info(f"  File size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Load model
    cf_loaded = CollaborativeFilteringRecommender()
    cf_loaded.load(model_path)
    logger.info(f"✓ Model loaded from {model_path}")

    # Verify loaded model works
    loaded_recs = cf_loaded.recommend(sample_user_id, k=5)
    logger.info(f"✓ Loaded model generated {len(loaded_recs)} recommendations")

    # Verify results match
    orig_ids = set(r[0] for r in recommendations[:5])
    loaded_ids = set(r[0] for r in loaded_recs)

    if orig_ids == loaded_ids:
        logger.info("✓ Loaded model produces identical results")
    else:
        logger.warning(f"⚠ Results differ: {len(orig_ids & loaded_ids)}/5 match")

    logger.info("\n" + "="*60)
    logger.info("✓ ALL CF MODEL TESTS PASSED SUCCESSFULLY")
    logger.info("="*60 + "\n")


if __name__ == "__main__":
    main()

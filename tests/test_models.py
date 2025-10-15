"""
Test script for recommendation models.
"""

from src.data_loader import DataLoader
from src.models import CollaborativeFilteringRecommender, NLPRecommender
from src.utils import setup_logger
import time

# Setup logger
logger = setup_logger("test_models")


def test_cf_recommender(users_df, likes_df):
    """Test collaborative filtering recommender."""
    logger.info("\n" + "="*60)
    logger.info("TESTING COLLABORATIVE FILTERING RECOMMENDER")
    logger.info("="*60)

    # Initialize and train
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
    logger.info("\n--- Testing Recommendations ---")

    # Get a user with decent activity
    active_users = likes_df['user_id'].value_counts()
    sample_user_id = active_users.index[10]  # User with moderate activity

    logger.info(f"Generating recommendations for user: {sample_user_id[:8]}...")

    start_time = time.time()
    recommendations = cf_model.recommend(sample_user_id, k=10)
    rec_time = time.time() - start_time

    logger.info(f"\n✓ Generated {len(recommendations)} recommendations in {rec_time*1000:.2f}ms")
    logger.info("\nTop 5 Recommendations:")
    for i, (rec_id, score) in enumerate(recommendations[:5], 1):
        logger.info(f"  {i}. User {rec_id[:8]}... (score: {score:.4f})")

    # Test scoring
    logger.info("\n--- Testing Scoring ---")
    candidate_ids = [rec[0] for rec in recommendations[:5]]
    scores = cf_model.score(sample_user_id, candidate_ids)

    logger.info(f"✓ Scored {len(scores)} candidates")
    for cid, score in list(scores.items())[:3]:
        logger.info(f"  User {cid[:8]}...: {score:.4f}")

    # Test cold-start user
    logger.info("\n--- Testing Cold-Start Handling ---")
    cold_start_users = likes_df['user_id'].value_counts()
    cold_user_id = cold_start_users[cold_start_users < 3].index[0]

    logger.info(f"Cold-start user: {cold_user_id[:8]}... (very few interactions)")
    cold_recs = cf_model.recommend(cold_user_id, k=5)
    logger.info(f"✓ Generated {len(cold_recs)} cold-start recommendations")

    # Test save/load
    logger.info("\n--- Testing Save/Load ---")
    from pathlib import Path
    model_path = Path("results/models/cf_test.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    cf_model.save(model_path)
    logger.info(f"✓ Model saved to {model_path}")

    # Load model
    cf_loaded = CollaborativeFilteringRecommender()
    cf_loaded.load(model_path)
    logger.info(f"✓ Model loaded from {model_path}")

    # Verify loaded model works
    loaded_recs = cf_loaded.recommend(sample_user_id, k=5)
    logger.info(f"✓ Loaded model generated {len(loaded_recs)} recommendations")

    return cf_model


def test_nlp_recommender(users_df):
    """Test NLP recommender."""
    logger.info("\n" + "="*60)
    logger.info("TESTING NLP RECOMMENDER")
    logger.info("="*60)

    # Initialize and train
    nlp_model = NLPRecommender(
        model_name='all-MiniLM-L6-v2',
        batch_size=64
    )

    logger.info("\nNote: This will download the sentence-transformer model if not cached")

    start_time = time.time()
    nlp_model.fit(users_df)
    train_time = time.time() - start_time

    logger.info(f"\n✓ Training completed in {train_time:.2f} seconds")

    # Test recommendations
    logger.info("\n--- Testing Recommendations ---")
    sample_user_id = users_df['user_id'].iloc[0]

    logger.info(f"Generating recommendations for user: {sample_user_id[:8]}...")
    logger.info(f"User profile: {nlp_model.profile_texts.loc[sample_user_id][:100]}...")

    start_time = time.time()
    recommendations = nlp_model.recommend(sample_user_id, k=10)
    rec_time = time.time() - start_time

    logger.info(f"\n✓ Generated {len(recommendations)} recommendations in {rec_time*1000:.2f}ms")
    logger.info("\nTop 5 Recommendations:")
    for i, (rec_id, score) in enumerate(recommendations[:5], 1):
        profile = nlp_model.profile_texts.loc[rec_id][:80]
        logger.info(f"  {i}. User {rec_id[:8]}... (score: {score:.4f})")
        logger.info(f"     Profile: {profile}...")

    # Test scoring
    logger.info("\n--- Testing Scoring ---")
    candidate_ids = [rec[0] for rec in recommendations[:5]]
    scores = nlp_model.score(sample_user_id, candidate_ids)

    logger.info(f"✓ Scored {len(scores)} candidates")
    for cid, score in list(scores.items())[:3]:
        logger.info(f"  User {cid[:8]}...: {score:.4f}")

    # Test similar profiles
    logger.info("\n--- Testing Similar Profiles ---")
    similar = nlp_model.get_similar_profiles(sample_user_id, k=3)

    logger.info(f"✓ Found {len(similar)} similar profiles:")
    for uid, text, score in similar:
        logger.info(f"  User {uid[:8]}... (similarity: {score:.4f})")
        logger.info(f"    {text[:80]}...")

    # Test save/load
    logger.info("\n--- Testing Save/Load ---")
    from pathlib import Path
    model_path = Path("results/models/nlp_test")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    nlp_model.save(model_path)
    logger.info(f"✓ Model saved to {model_path}")

    # Load model
    nlp_loaded = NLPRecommender()
    nlp_loaded.load(model_path)
    logger.info(f"✓ Model loaded from {model_path}")

    # Verify loaded model works
    loaded_recs = nlp_loaded.recommend(sample_user_id, k=5)
    logger.info(f"✓ Loaded model generated {len(loaded_recs)} recommendations")

    return nlp_model


def compare_models(cf_model, nlp_model, users_df, likes_df):
    """Compare recommendations from both models."""
    logger.info("\n" + "="*60)
    logger.info("COMPARING MODELS")
    logger.info("="*60)

    # Get a user with good activity
    active_users = likes_df['user_id'].value_counts()
    sample_user_id = active_users.index[5]

    user_info = users_df[users_df['user_id'] == sample_user_id].iloc[0]
    logger.info(f"\nTest User: {sample_user_id[:8]}...")
    logger.info(f"  Gender: {user_info['gender']}")
    logger.info(f"  Age: {user_info['age']}")
    logger.info(f"  City: {user_info['city']}")
    logger.info(f"  Job: {user_info['job_title']}")
    logger.info(f"  Profile: {nlp_model.profile_texts.loc[sample_user_id][:100]}...")

    # Get recommendations from both
    k = 10
    cf_recs = cf_model.recommend(sample_user_id, k=k)
    nlp_recs = nlp_model.recommend(sample_user_id, k=k)

    logger.info(f"\n--- CF Recommendations (Top {k}) ---")
    for i, (uid, score) in enumerate(cf_recs, 1):
        user = users_df[users_df['user_id'] == uid].iloc[0]
        logger.info(
            f"{i}. {uid[:8]}... | {user['gender']} {user['age']} | "
            f"{user['city']} | {user['job_title']} | Score: {score:.4f}"
        )

    logger.info(f"\n--- NLP Recommendations (Top {k}) ---")
    for i, (uid, score) in enumerate(nlp_recs, 1):
        user = users_df[users_df['user_id'] == uid].iloc[0]
        logger.info(
            f"{i}. {uid[:8]}... | {user['gender']} {user['age']} | "
            f"{user['city']} | {user['job_title']} | Score: {score:.4f}"
        )

    # Check overlap
    cf_ids = set(r[0] for r in cf_recs)
    nlp_ids = set(r[0] for r in nlp_recs)
    overlap = cf_ids & nlp_ids

    logger.info(f"\n--- Overlap Analysis ---")
    logger.info(f"CF recommended: {len(cf_ids)} users")
    logger.info(f"NLP recommended: {len(nlp_ids)} users")
    logger.info(f"Overlap: {len(overlap)} users ({len(overlap)/k*100:.1f}%)")
    if overlap:
        logger.info(f"Common recommendations: {[uid[:8] for uid in list(overlap)[:3]]}...")


def main():
    """Main test function."""
    logger.info("="*60)
    logger.info("RECOMMENDATION MODELS TEST SUITE")
    logger.info("="*60)

    # Load data
    logger.info("\nLoading data...")
    loader = DataLoader()
    users_df, likes_df, matches_df = loader.load_all()
    logger.info(f"✓ Data loaded: {len(users_df)} users, {len(likes_df)} likes")

    # Test CF Recommender
    cf_model = test_cf_recommender(users_df, likes_df)

    # Test NLP Recommender
    nlp_model = test_nlp_recommender(users_df)

    # Compare models
    compare_models(cf_model, nlp_model, users_df, likes_df)

    logger.info("\n" + "="*60)
    logger.info("✓ ALL MODEL TESTS COMPLETED SUCCESSFULLY")
    logger.info("="*60 + "\n")


if __name__ == "__main__":
    main()

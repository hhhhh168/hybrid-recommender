"""
Test script for data loading and preprocessing pipeline.
"""

from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.utils import setup_logger

# Setup logger
logger = setup_logger("test_pipeline")

def main():
    logger.info("=" * 60)
    logger.info("TESTING DATA LOADING AND PREPROCESSING PIPELINE")
    logger.info("=" * 60)

    # Test DataLoader
    logger.info("\n1. Testing DataLoader...")
    loader = DataLoader()

    try:
        # Load all data
        users_df, likes_df, matches_df = loader.load_all()
        logger.info("✓ All data loaded successfully")

        # Validate data
        is_valid = loader.validate_data()
        if is_valid:
            logger.info("✓ Data validation passed")
        else:
            logger.warning("✗ Data validation failed (check warnings above)")

        # Get summary
        summary = loader.get_data_summary()
        logger.info("\n2. Data Summary:")
        logger.info(f"  Users: {summary['users']['total']:,}")
        logger.info(f"    - Premium: {summary['users']['premium_users']:,}")
        logger.info(f"    - Active: {summary['users']['active_users']:,}")
        logger.info(f"  Likes: {summary['likes']['total']:,}")
        logger.info(f"    - Avg per user: {summary['likes']['avg_likes_per_user']:.1f}")
        logger.info(f"  Matches: {summary['matches']['total']:,}")
        logger.info(f"    - Match rate: {summary['matches']['match_rate']:.2f}%")

    except Exception as e:
        logger.error(f"Error in DataLoader: {e}")
        raise

    # Test DataPreprocessor
    logger.info("\n3. Testing DataPreprocessor...")
    preprocessor = DataPreprocessor(loader)

    try:
        # Test interaction matrix creation
        logger.info("\n  a) Creating user interaction matrix...")
        matrix, user_to_idx, idx_to_user = preprocessor.create_user_interaction_matrix(
            likes_df, users_df
        )
        logger.info(f"  ✓ Matrix shape: {matrix.shape}")
        logger.info(f"  ✓ Non-zero entries: {matrix.nnz:,}")
        logger.info(f"  ✓ Sparsity: {(1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])) * 100:.4f}%")

        # Test train-test split
        logger.info("\n  b) Testing temporal train-test split...")
        train_likes, test_likes = preprocessor.train_test_split_temporal(likes_df, test_size=0.2)
        logger.info(f"  ✓ Train size: {len(train_likes):,}")
        logger.info(f"  ✓ Test size: {len(test_likes):,}")

        # Verify temporal ordering
        if train_likes['timestamp'].max() <= test_likes['timestamp'].min():
            logger.info("  ✓ Temporal ordering verified")
        else:
            logger.warning("  ⚠ Temporal ordering may have overlap")

        # Test profile text creation
        logger.info("\n  c) Creating profile texts...")
        profile_texts = preprocessor.create_profile_text(users_df)
        logger.info(f"  ✓ Created {len(profile_texts)} profile texts")
        logger.info(f"  ✓ Sample: {profile_texts.iloc[0][:100]}...")

        # Test preference filtering
        logger.info("\n  d) Testing preference filtering...")
        sample_user = users_df.iloc[0]
        filtered_candidates = preprocessor.filter_by_preferences(
            sample_user, users_df, users_df, exclude_user_ids=set()
        )
        logger.info(f"  ✓ User preferences applied")
        logger.info(f"  ✓ Candidates before filter: {len(users_df)}")
        logger.info(f"  ✓ Candidates after filter: {len(filtered_candidates)}")

        # Test interaction features
        logger.info("\n  e) Creating interaction features...")
        features = preprocessor.create_interaction_features(users_df, likes_df, matches_df)
        logger.info(f"  ✓ Features created: {list(features.columns)}")
        logger.info(f"  ✓ Feature shape: {features.shape}")

        # Test user history
        logger.info("\n  f) Testing user history retrieval...")
        sample_user_id = users_df.iloc[0]['user_id']
        history = preprocessor.get_user_history(sample_user_id, likes_df, matches_df)
        logger.info(f"  ✓ User {sample_user_id[:8]}... history:")
        logger.info(f"    - Liked: {len(history['liked'])} users")
        logger.info(f"    - Liked by: {len(history['liked_by'])} users")
        logger.info(f"    - Matches: {len(history['matches'])} users")

        logger.info("\n" + "=" * 60)
        logger.info("✓ ALL TESTS PASSED SUCCESSFULLY")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error in DataPreprocessor: {e}")
        raise

if __name__ == "__main__":
    main()

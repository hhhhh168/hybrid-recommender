"""
Data preprocessing module for WorkHeart recommendation system.

This module handles feature engineering, train-test splits, and data transformations
for the hybrid recommendation system.
"""

import pandas as pd
import numpy as np
from scipy import sparse
from datetime import datetime
from typing import Tuple, Optional, Dict, Set
import warnings

from src.utils import setup_logger
from src.data_loader import DataLoader

# Initialize logger
logger = setup_logger(__name__)


class DataPreprocessor:
    """
    Preprocess and transform data for recommendation models.

    This class handles creating interaction matrices, temporal train-test splits,
    text feature generation, and preference-based filtering.
    """

    def __init__(self, data_loader: Optional[DataLoader] = None):
        """
        Initialize DataPreprocessor.

        Args:
            data_loader: Optional DataLoader instance with loaded data
        """
        self.data_loader = data_loader
        logger.info("DataPreprocessor initialized")

    def create_user_interaction_matrix(
        self,
        likes_df: pd.DataFrame,
        users_df: Optional[pd.DataFrame] = None
    ) -> Tuple[sparse.csr_matrix, Dict[str, int], Dict[int, str]]:
        """
        Create a sparse user-user interaction matrix from likes data.

        Args:
            likes_df: DataFrame with columns ['user_id', 'liked_user_id', 'action']
            users_df: Optional DataFrame with user data to ensure all users are included

        Returns:
            Tuple of:
                - Sparse CSR matrix (users x users) with interaction weights
                - user_id_to_idx: Dict mapping user_id to matrix index
                - idx_to_user_id: Dict mapping matrix index to user_id
        """
        logger.info("Creating user interaction matrix...")

        # Get all unique user IDs
        if users_df is not None:
            all_user_ids = users_df['user_id'].unique()
        else:
            all_user_ids = pd.concat([
                likes_df['user_id'],
                likes_df['liked_user_id']
            ]).unique()

        n_users = len(all_user_ids)

        # Create mappings
        user_id_to_idx = {user_id: idx for idx, user_id in enumerate(all_user_ids)}
        idx_to_user_id = {idx: user_id for user_id, idx in user_id_to_idx.items()}

        logger.info(f"Matrix dimensions: {n_users} x {n_users}")

        # Prepare data for sparse matrix construction
        rows = []
        cols = []
        data = []

        # Define weights for different actions
        action_weights = {
            'like': 2.0,
            'superlike': 3.0
        }

        for _, like in likes_df.iterrows():
            user_id = like['user_id']
            liked_id = like['liked_user_id']
            action = like['action']

            # Get matrix indices
            if user_id in user_id_to_idx and liked_id in user_id_to_idx:
                user_idx = user_id_to_idx[user_id]
                liked_idx = user_id_to_idx[liked_id]

                # Get weight for this action
                weight = action_weights.get(action, 1.0)

                rows.append(user_idx)
                cols.append(liked_idx)
                data.append(weight)

        # Create sparse matrix
        interaction_matrix = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(n_users, n_users),
            dtype=np.float32
        )

        # Log statistics
        nnz = interaction_matrix.nnz
        total_elements = n_users * n_users
        sparsity = (1 - nnz / total_elements) * 100

        logger.info(f"Matrix created with {nnz:,} non-zero entries")
        logger.info(f"Sparsity: {sparsity:.4f}%")
        logger.info(f"Memory: {interaction_matrix.data.nbytes / 1024 / 1024:.2f} MB")

        return interaction_matrix, user_id_to_idx, idx_to_user_id

    def train_test_split_temporal(
        self,
        likes_df: pd.DataFrame,
        test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split likes data into train and test sets based on temporal ordering.

        Args:
            likes_df: DataFrame with likes data (must have 'timestamp' column)
            test_size: Fraction of data to use for testing (default: 0.2)

        Returns:
            Tuple of (train_likes_df, test_likes_df)
        """
        logger.info(f"Performing temporal train-test split (test_size={test_size})...")

        if 'timestamp' not in likes_df.columns:
            raise ValueError("likes_df must have 'timestamp' column for temporal split")

        # Sort by timestamp
        sorted_likes = likes_df.sort_values('timestamp').reset_index(drop=True)

        # Calculate split index
        split_idx = int(len(sorted_likes) * (1 - test_size))

        # Split
        train_df = sorted_likes.iloc[:split_idx].copy()
        test_df = sorted_likes.iloc[split_idx:].copy()

        # Validate temporal ordering
        train_max_time = train_df['timestamp'].max()
        test_min_time = test_df['timestamp'].min()

        logger.info(f"Train set: {len(train_df):,} likes")
        logger.info(f"  Date range: {train_df['timestamp'].min()} to {train_max_time}")
        logger.info(f"Test set: {len(test_df):,} likes")
        logger.info(f"  Date range: {test_min_time} to {test_df['timestamp'].max()}")

        if train_max_time >= test_min_time:
            logger.warning(
                f"Warning: Train max time ({train_max_time}) >= "
                f"Test min time ({test_min_time}). Temporal ordering may overlap."
            )

        return train_df, test_df

    def create_profile_text(self, users_df: pd.DataFrame) -> pd.Series:
        """
        Create combined profile text from user attributes.

        Combines bio, job title, school, and city into a single text field
        suitable for NLP processing.

        Args:
            users_df: DataFrame with user data

        Returns:
            Series with user_id as index and profile text as values
        """
        logger.info("Creating profile text for users...")

        def combine_profile(row: pd.Series) -> str:
            """Combine user profile fields into text."""
            import re

            parts = []

            # Add bio if available
            if pd.notna(row.get('bio')) and str(row['bio']).strip():
                bio_text = str(row['bio']).strip()
                # Limit bio length to prevent token overflow (max ~800 chars)
                if len(bio_text) > 800:
                    bio_text = bio_text[:800] + "..."
                parts.append(bio_text)

            # Add job title
            if pd.notna(row.get('job_title')):
                parts.append(str(row['job_title']))

            # Add school
            if pd.notna(row.get('school')):
                parts.append(f"at {row['school']}")

            # Add city
            if pd.notna(row.get('city')):
                parts.append(f"in {row['city']}")

            # Combine all parts
            text = " ".join(parts)

            # Clean text
            text = text.lower().strip()

            # Remove URLs (simple pattern)
            text = re.sub(r'http\S+|www\.\S+', '', text)

            # Normalize whitespace (multiple spaces/newlines to single space)
            text = re.sub(r'\s+', ' ', text)

            # Remove excessive punctuation (more than 3 repeating)
            text = re.sub(r'([!?.]{3,})', '...', text)

            # Final strip
            text = text.strip()

            # Ensure non-empty
            if not text:
                text = "no profile information"

            return text

        # Apply to all users
        profile_texts = users_df.apply(combine_profile, axis=1)

        # Set index to user_id
        if 'user_id' in users_df.columns:
            profile_texts.index = users_df['user_id']

        logger.info(f"Created profile text for {len(profile_texts)} users")
        logger.info(f"Avg text length: {profile_texts.str.len().mean():.1f} characters")

        return profile_texts

    def filter_by_preferences(
        self,
        user: pd.Series,
        candidates_df: pd.DataFrame,
        users_df: pd.DataFrame,
        exclude_user_ids: Optional[Set[str]] = None
    ) -> pd.DataFrame:
        """
        Filter candidate users based on user preferences and exclusions.

        Args:
            user: Series representing the target user (from users_df)
            candidates_df: DataFrame of candidate users to filter
            users_df: Full users DataFrame (for reference)
            exclude_user_ids: Optional set of user IDs to exclude
                             (e.g., already liked/matched users)

        Returns:
            Filtered DataFrame of candidates matching preferences
        """
        logger.debug(f"Filtering candidates for user {user.get('user_id', 'unknown')}")

        filtered = candidates_df.copy()

        # Remove the user themselves
        if 'user_id' in user.index:
            filtered = filtered[filtered['user_id'] != user['user_id']]

        # Exclude specified users (already liked/matched)
        if exclude_user_ids:
            filtered = filtered[~filtered['user_id'].isin(exclude_user_ids)]

        # Gender preference filter
        preferred_gender = user.get('matching_pref_gender')
        if pd.notna(preferred_gender) and preferred_gender != 'both':
            filtered = filtered[filtered['gender'] == preferred_gender]

        # Age preference filter (±10 years from user's preferred age)
        if pd.notna(user.get('matching_pref_age_min')) and pd.notna(user.get('matching_pref_age_max')):
            age_min = user['matching_pref_age_min']
            age_max = user['matching_pref_age_max']

            filtered = filtered[
                (filtered['age'] >= age_min) &
                (filtered['age'] <= age_max)
            ]

        # Location preference filter
        use_location = user.get('matching_pref_use_location', False)
        if use_location and pd.notna(user.get('city')):
            filtered = filtered[filtered['city'] == user['city']]

        logger.debug(f"Filtered to {len(filtered)} candidates")

        return filtered

    @staticmethod
    def get_user_age(birthday_str: str) -> int:
        """
        Calculate user age from birthday string.

        Args:
            birthday_str: Birthday in MM/DD/YYYY format

        Returns:
            Age in years
        """
        try:
            birth_date = datetime.strptime(birthday_str, '%m/%d/%Y')
            today = datetime.now()
            age = today.year - birth_date.year - (
                (today.month, today.day) < (birth_date.month, birth_date.day)
            )
            return age
        except (ValueError, AttributeError, TypeError):
            logger.warning(f"Could not parse birthday: {birthday_str}")
            return 0

    def create_interaction_features(
        self,
        users_df: pd.DataFrame,
        likes_df: pd.DataFrame,
        matches_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Create user-level interaction features for modeling.

        Args:
            users_df: User data
            likes_df: Likes data
            matches_df: Optional matches data

        Returns:
            DataFrame with user_id and engineered features
        """
        logger.info("Creating interaction features...")

        features = users_df[['user_id']].copy()

        # Likes given
        likes_given = likes_df.groupby('user_id').size().rename('likes_given')
        features = features.merge(likes_given, left_on='user_id', right_index=True, how='left')

        # Likes received
        likes_received = likes_df.groupby('liked_user_id').size().rename('likes_received')
        features = features.merge(likes_received, left_on='user_id', right_index=True, how='left')

        # Superlike ratio (if applicable)
        if 'action' in likes_df.columns:
            superlikes = likes_df[likes_df['action'] == 'superlike'].groupby('user_id').size()
            features = features.merge(superlikes.rename('superlikes_given'), left_on='user_id', right_index=True, how='left')

            features['superlike_ratio'] = (
                features['superlikes_given'].fillna(0) /
                features['likes_given'].fillna(1)
            )

        # Match features
        if matches_df is not None:
            # Count matches per user
            matches1 = matches_df.groupby('user1_id').size()
            matches2 = matches_df.groupby('user2_id').size()
            total_matches = matches1.add(matches2, fill_value=0).rename('total_matches')

            features = features.merge(total_matches, left_on='user_id', right_index=True, how='left')

            # Match rate
            features['match_rate'] = (
                features['total_matches'].fillna(0) /
                (features['likes_given'].fillna(0) + features['likes_received'].fillna(0) + 1)
            )

        # Fill NaN values
        features = features.fillna(0)

        logger.info(f"Created {len(features.columns) - 1} interaction features")

        return features

    def get_user_history(
        self,
        user_id: str,
        likes_df: pd.DataFrame,
        matches_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Set[str]]:
        """
        Get user's interaction history.

        Args:
            user_id: User ID to get history for
            likes_df: Likes data
            matches_df: Optional matches data

        Returns:
            Dictionary with sets of:
                - 'liked': Users this user has liked
                - 'liked_by': Users who liked this user
                - 'matches': Users this user matched with
        """
        history = {
            'liked': set(),
            'liked_by': set(),
            'matches': set()
        }

        # Users this user liked
        history['liked'] = set(likes_df[likes_df['user_id'] == user_id]['liked_user_id'])

        # Users who liked this user
        history['liked_by'] = set(likes_df[likes_df['liked_user_id'] == user_id]['user_id'])

        # Matches
        if matches_df is not None:
            matches1 = set(matches_df[matches_df['user1_id'] == user_id]['user2_id'])
            matches2 = set(matches_df[matches_df['user2_id'] == user_id]['user1_id'])
            history['matches'] = matches1 | matches2

        return history

    def normalize_interaction_matrix(
        self,
        interaction_matrix: sparse.csr_matrix,
        method: str = 'l2'
    ) -> sparse.csr_matrix:
        """
        Normalize interaction matrix rows.

        Args:
            interaction_matrix: Sparse interaction matrix
            method: Normalization method ('l2', 'l1', or 'max')

        Returns:
            Normalized sparse matrix
        """
        logger.info(f"Normalizing interaction matrix using {method} normalization...")

        from sklearn.preprocessing import normalize

        normalized = normalize(interaction_matrix, norm=method, axis=1)

        logger.info("Matrix normalized")

        return normalized

    def create_user_similarity_matrix(
        self,
        interaction_matrix: sparse.csr_matrix,
        metric: str = 'cosine',
        k: int = 50
    ) -> sparse.csr_matrix:
        """
        Create user-user similarity matrix.

        Args:
            interaction_matrix: User interaction matrix (users x users)
            metric: Similarity metric ('cosine', 'jaccard')
            k: Number of nearest neighbors to keep (for sparsity)

        Returns:
            Sparse similarity matrix
        """
        logger.info(f"Computing user similarity matrix using {metric} metric...")

        from sklearn.metrics.pairwise import cosine_similarity

        if metric == 'cosine':
            # Compute cosine similarity
            similarity = cosine_similarity(interaction_matrix, dense_output=False)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        logger.info(f"Similarity matrix computed: {similarity.shape}")

        # Keep only top-k similar users per user for efficiency
        if k is not None and k < similarity.shape[1]:
            logger.info(f"Keeping top-{k} neighbors per user...")

            # Convert to dense for easier manipulation
            similarity_dense = similarity.toarray()

            # For each row, keep only top-k values
            for i in range(similarity_dense.shape[0]):
                row = similarity_dense[i]
                # Get indices of top-k (excluding self)
                top_k_indices = np.argpartition(row, -k-1)[-k-1:]
                # Zero out all others
                mask = np.ones(len(row), dtype=bool)
                mask[top_k_indices] = False
                row[mask] = 0
                # Zero out self-similarity
                row[i] = 0

            similarity = sparse.csr_matrix(similarity_dense)

        logger.info(f"Similarity matrix sparsity: {(1 - similarity.nnz / (similarity.shape[0] * similarity.shape[1])) * 100:.2f}%")

        return similarity

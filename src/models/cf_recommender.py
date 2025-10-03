"""
Collaborative filtering recommender using user-user similarity.

This module implements a user-based collaborative filtering approach using
sparse matrix operations for efficient computation on large datasets.
"""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict, Optional, Set
from pathlib import Path
import pickle

from src.models.base_recommender import BaseRecommender
from src.preprocessing import DataPreprocessor
from src.utils import setup_logger

logger = setup_logger(__name__)


class CollaborativeFilteringRecommender(BaseRecommender):
    """
    User-based collaborative filtering recommender.

    Uses sparse user-user interaction matrix and cosine similarity to
    recommend users based on similar users' preferences.
    """

    def __init__(
        self,
        n_similar_users: int = 50,
        min_interactions: int = 5,
        like_weight: float = 2.0,
        superlike_weight: float = 3.0
    ):
        """
        Initialize collaborative filtering recommender.

        Args:
            n_similar_users: Number of similar users to consider for recommendations
            min_interactions: Minimum interactions before using CF (else cold-start)
            like_weight: Weight for 'like' actions
            superlike_weight: Weight for 'superlike' actions
        """
        super().__init__(name="CollaborativeFilteringRecommender")
        self.n_similar_users = n_similar_users
        self.min_interactions = min_interactions
        self.like_weight = like_weight
        self.superlike_weight = superlike_weight

        # Model components (set during fit)
        self.interaction_matrix: Optional[csr_matrix] = None
        self.similarity_matrix: Optional[csr_matrix] = None
        self.user_id_to_idx: Optional[Dict[str, int]] = None
        self.idx_to_user_id: Optional[Dict[int, str]] = None
        self.users_df: Optional[pd.DataFrame] = None
        self.preprocessor: Optional[DataPreprocessor] = None
        self.popular_users: Optional[List[str]] = None

    def fit(
        self,
        likes_df: pd.DataFrame,
        users_df: pd.DataFrame
    ) -> 'CollaborativeFilteringRecommender':
        """
        Train the collaborative filtering model.

        Args:
            likes_df: DataFrame with likes data (columns: user_id, liked_user_id, action)
            users_df: DataFrame with user data

        Returns:
            Self for method chaining
        """
        logger.info(f"Training {self.name}...")
        logger.info(f"Data: {len(users_df)} users, {len(likes_df)} likes")

        self.preprocessor = DataPreprocessor()
        self.users_df = users_df.copy()

        # Create interaction matrix with action weights
        logger.info("Creating user interaction matrix...")
        self.interaction_matrix, self.user_id_to_idx, self.idx_to_user_id = \
            self._create_weighted_interaction_matrix(likes_df, users_df)

        # Compute user-user similarity
        logger.info("Computing user-user similarity matrix...")
        self.similarity_matrix = self._compute_similarity_matrix()

        # Identify popular users for cold-start recommendations
        logger.info("Identifying popular users for cold-start...")
        self.popular_users = self._get_popular_users(likes_df, k=100)

        self.is_trained = True
        logger.info(f"✓ {self.name} training complete")

        return self

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        exclude_ids: Optional[Set[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Generate top-K recommendations for a user.

        Args:
            user_id: ID of the user to generate recommendations for
            k: Number of recommendations to return
            exclude_ids: Optional set of user IDs to exclude

        Returns:
            List of (user_id, score) tuples, sorted by score descending

        Raises:
            ValueError: If model not trained or user_id invalid
        """
        self._check_trained()

        if user_id not in self.user_id_to_idx:
            raise ValueError(f"User {user_id} not found in training data")

        # Check for cold-start
        user_idx = self.user_id_to_idx[user_id]
        n_interactions = self.interaction_matrix[user_idx].nnz

        if n_interactions < self.min_interactions:
            logger.warning(
                f"Cold-start: User {user_id} has only {n_interactions} interactions. "
                f"Using popular user recommendations."
            )
            return self._recommend_popular(user_id, k, exclude_ids)

        # Get similar users
        similar_users = self._get_similar_users(user_id, k=self.n_similar_users)

        if not similar_users:
            logger.warning(f"No similar users found for {user_id}. Using popular recommendations.")
            return self._recommend_popular(user_id, k, exclude_ids)

        # Aggregate recommendations from similar users
        candidate_scores = self._aggregate_from_similar_users(
            user_id,
            similar_users,
            exclude_ids
        )

        # Apply user preferences filter
        candidate_scores = self._apply_preference_filter(user_id, candidate_scores)

        # Sort and return top-K
        recommendations = sorted(
            candidate_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]

        logger.debug(f"Generated {len(recommendations)} recommendations for user {user_id}")

        return recommendations

    def score(
        self,
        user_id: str,
        candidate_ids: List[str]
    ) -> Dict[str, float]:
        """
        Compute recommendation scores for specific candidates.

        Args:
            user_id: ID of the user
            candidate_ids: List of candidate user IDs to score

        Returns:
            Dictionary mapping candidate_id to score
        """
        self._check_trained()

        if user_id not in self.user_id_to_idx:
            raise ValueError(f"User {user_id} not found in training data")

        # Get similar users
        similar_users = self._get_similar_users(user_id, k=self.n_similar_users)

        if not similar_users:
            return {cid: 0.0 for cid in candidate_ids}

        # Score each candidate
        scores = {}
        for candidate_id in candidate_ids:
            if candidate_id not in self.user_id_to_idx:
                scores[candidate_id] = 0.0
                continue

            candidate_idx = self.user_id_to_idx[candidate_id]

            # Compute weighted score based on similar users
            score = 0.0
            for sim_user_id, similarity in similar_users:
                sim_user_idx = self.user_id_to_idx[sim_user_id]

                # Check if similar user liked this candidate
                interaction = self.interaction_matrix[sim_user_idx, candidate_idx]

                if interaction > 0:
                    score += similarity * interaction

            scores[candidate_id] = score

        return scores

    def save(self, filepath: Path) -> None:
        """
        Save the trained model to disk.

        Args:
            filepath: Path to save the model
        """
        self._check_trained()

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'name': self.name,
            'n_similar_users': self.n_similar_users,
            'min_interactions': self.min_interactions,
            'like_weight': self.like_weight,
            'superlike_weight': self.superlike_weight,
            'interaction_matrix': self.interaction_matrix,
            'similarity_matrix': self.similarity_matrix,
            'user_id_to_idx': self.user_id_to_idx,
            'idx_to_user_id': self.idx_to_user_id,
            'users_df': self.users_df,
            'popular_users': self.popular_users,
            'is_trained': self.is_trained
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: Path) -> 'CollaborativeFilteringRecommender':
        """
        Load a trained model from disk.

        Args:
            filepath: Path to load the model from

        Returns:
            Self for method chaining
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.name = model_data['name']
        self.n_similar_users = model_data['n_similar_users']
        self.min_interactions = model_data['min_interactions']
        self.like_weight = model_data['like_weight']
        self.superlike_weight = model_data['superlike_weight']
        self.interaction_matrix = model_data['interaction_matrix']
        self.similarity_matrix = model_data['similarity_matrix']
        self.user_id_to_idx = model_data['user_id_to_idx']
        self.idx_to_user_id = model_data['idx_to_user_id']
        self.users_df = model_data['users_df']
        self.popular_users = model_data['popular_users']
        self.is_trained = model_data['is_trained']

        logger.info(f"Model loaded from {filepath}")

        return self

    def _create_weighted_interaction_matrix(
        self,
        likes_df: pd.DataFrame,
        users_df: pd.DataFrame
    ) -> Tuple[csr_matrix, Dict[str, int], Dict[int, str]]:
        """
        Create weighted user-user interaction matrix.

        Args:
            likes_df: Likes data
            users_df: User data

        Returns:
            Tuple of (sparse matrix, user_id_to_idx, idx_to_user_id)
        """
        all_user_ids = users_df['user_id'].unique()
        n_users = len(all_user_ids)

        # Create mappings
        user_id_to_idx = {uid: idx for idx, uid in enumerate(all_user_ids)}
        idx_to_user_id = {idx: uid for uid, idx in user_id_to_idx.items()}

        # Build sparse matrix
        rows, cols, data = [], [], []

        for _, like in likes_df.iterrows():
            user_id = like['user_id']
            liked_id = like['liked_user_id']
            action = like['action']

            if user_id in user_id_to_idx and liked_id in user_id_to_idx:
                user_idx = user_id_to_idx[user_id]
                liked_idx = user_id_to_idx[liked_id]

                # Apply weights
                weight = self.superlike_weight if action == 'superlike' else self.like_weight

                rows.append(user_idx)
                cols.append(liked_idx)
                data.append(weight)

        matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(n_users, n_users),
            dtype=np.float32
        )

        logger.info(f"Interaction matrix: {matrix.shape}, {matrix.nnz:,} non-zero entries")

        return matrix, user_id_to_idx, idx_to_user_id

    def _compute_similarity_matrix(self) -> csr_matrix:
        """
        Compute user-user cosine similarity matrix.

        Returns:
            Sparse similarity matrix
        """
        # Normalize interaction matrix (L2 norm)
        from sklearn.preprocessing import normalize
        normalized = normalize(self.interaction_matrix, norm='l2', axis=1)

        # Compute cosine similarity
        similarity = cosine_similarity(normalized, dense_output=False)

        # Zero out self-similarity
        similarity.setdiag(0)

        logger.info(f"Similarity matrix: {similarity.shape}, {similarity.nnz:,} non-zero entries")

        return similarity

    def _get_similar_users(
        self,
        user_id: str,
        k: int = 50
    ) -> List[Tuple[str, float]]:
        """
        Get top-K most similar users.

        Args:
            user_id: Target user ID
            k: Number of similar users to return

        Returns:
            List of (user_id, similarity_score) tuples
        """
        user_idx = self.user_id_to_idx[user_id]

        # Get similarity scores for this user
        similarity_row = self.similarity_matrix[user_idx].toarray().flatten()

        # Get top-k indices (excluding self)
        top_k_indices = np.argpartition(similarity_row, -k)[-k:]
        top_k_indices = top_k_indices[similarity_row[top_k_indices] > 0]  # Only positive similarities

        # Sort by similarity
        top_k_indices = top_k_indices[np.argsort(similarity_row[top_k_indices])[::-1]]

        # Convert to user IDs with scores
        similar_users = [
            (self.idx_to_user_id[idx], similarity_row[idx])
            for idx in top_k_indices
        ]

        return similar_users

    def _aggregate_from_similar_users(
        self,
        user_id: str,
        similar_users: List[Tuple[str, float]],
        exclude_ids: Optional[Set[str]] = None
    ) -> Dict[str, float]:
        """
        Aggregate candidate scores from similar users.

        Args:
            user_id: Target user ID
            similar_users: List of (user_id, similarity) tuples
            exclude_ids: User IDs to exclude

        Returns:
            Dictionary of candidate_id -> score
        """
        if exclude_ids is None:
            exclude_ids = set()

        # Add target user to exclusions
        exclude_ids = exclude_ids | {user_id}

        candidate_scores = {}

        for sim_user_id, similarity in similar_users:
            sim_user_idx = self.user_id_to_idx[sim_user_id]

            # Get all users this similar user liked
            liked_indices = self.interaction_matrix[sim_user_idx].nonzero()[1]

            for liked_idx in liked_indices:
                liked_id = self.idx_to_user_id[liked_idx]

                # Skip if excluded
                if liked_id in exclude_ids:
                    continue

                # Get interaction weight
                interaction = self.interaction_matrix[sim_user_idx, liked_idx]

                # Add weighted score
                score = similarity * interaction
                candidate_scores[liked_id] = candidate_scores.get(liked_id, 0.0) + score

        return candidate_scores

    def _apply_preference_filter(
        self,
        user_id: str,
        candidate_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Filter candidates by user preferences.

        Args:
            user_id: Target user ID
            candidate_scores: Dictionary of candidate_id -> score

        Returns:
            Filtered dictionary of candidate_id -> score
        """
        # Get user preferences
        user = self.users_df[self.users_df['user_id'] == user_id].iloc[0]

        filtered_scores = {}

        for candidate_id, score in candidate_scores.items():
            # Get candidate info
            candidate = self.users_df[self.users_df['user_id'] == candidate_id]

            if candidate.empty:
                continue

            candidate = candidate.iloc[0]

            # Gender filter
            preferred_gender = user.get('matching_pref_gender')
            if pd.notna(preferred_gender) and preferred_gender != 'both':
                if candidate['gender'] != preferred_gender:
                    continue

            # Age filter
            if pd.notna(user.get('matching_pref_age_min')) and pd.notna(user.get('matching_pref_age_max')):
                if not (user['matching_pref_age_min'] <= candidate['age'] <= user['matching_pref_age_max']):
                    continue

            # Location filter
            if user.get('matching_pref_use_location', False):
                if candidate['city'] != user['city']:
                    continue

            filtered_scores[candidate_id] = score

        return filtered_scores

    def _get_popular_users(
        self,
        likes_df: pd.DataFrame,
        k: int = 100
    ) -> List[str]:
        """
        Get most popular users (most liked).

        Args:
            likes_df: Likes data
            k: Number of popular users to return

        Returns:
            List of user IDs
        """
        popularity = likes_df['liked_user_id'].value_counts().head(k)
        return popularity.index.tolist()

    def _recommend_popular(
        self,
        user_id: str,
        k: int = 10,
        exclude_ids: Optional[Set[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Recommend popular users (cold-start fallback).

        Args:
            user_id: Target user ID
            k: Number of recommendations
            exclude_ids: User IDs to exclude

        Returns:
            List of (user_id, score) tuples
        """
        if exclude_ids is None:
            exclude_ids = set()

        exclude_ids = exclude_ids | {user_id}

        # Get user preferences
        user = self.users_df[self.users_df['user_id'] == user_id].iloc[0]

        # Filter popular users by preferences
        recommendations = []
        for pop_user_id in self.popular_users:
            if pop_user_id in exclude_ids:
                continue

            # Check preferences
            candidate = self.users_df[self.users_df['user_id'] == pop_user_id]
            if candidate.empty:
                continue

            candidate = candidate.iloc[0]

            # Apply filters
            preferred_gender = user.get('matching_pref_gender')
            if pd.notna(preferred_gender) and preferred_gender != 'both':
                if candidate['gender'] != preferred_gender:
                    continue

            if pd.notna(user.get('matching_pref_age_min')) and pd.notna(user.get('matching_pref_age_max')):
                if not (user['matching_pref_age_min'] <= candidate['age'] <= user['matching_pref_age_max']):
                    continue

            if user.get('matching_pref_use_location', False):
                if candidate['city'] != user['city']:
                    continue

            # Add with decreasing score
            score = 1.0 / (len(recommendations) + 1)
            recommendations.append((pop_user_id, score))

            if len(recommendations) >= k:
                break

        return recommendations

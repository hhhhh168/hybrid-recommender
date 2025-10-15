"""
Hybrid recommender combining collaborative filtering and NLP-based recommendations.

This module implements an adaptive hybrid approach that weights CF and NLP models
based on user activity level for optimal recommendation quality.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Set
from pathlib import Path
import pickle

from src.models.base_recommender import BaseRecommender
from src.models.cf_recommender import CollaborativeFilteringRecommender
from src.utils import setup_logger

# Lazy import to avoid loading sentence-transformers unless needed
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.models.nlp_recommender import NLPRecommender

logger = setup_logger(__name__)


class HybridRecommender(BaseRecommender):
    """
    Hybrid recommendation system combining CF and NLP approaches.

    Uses adaptive weighting based on user activity:
    - New users (< 10 likes): Rely more on NLP/content similarity
    - Moderate users (10-50 likes): Balanced CF + NLP
    - Active users (> 50 likes): Rely more on CF/behavioral patterns
    """

    def __init__(
        self,
        cf_model: Optional[CollaborativeFilteringRecommender] = None,
        nlp_model: Optional['NLPRecommender'] = None,
        default_alpha: float = 0.6
    ):
        """
        Initialize hybrid recommender.

        Args:
            cf_model: Pre-trained collaborative filtering model (optional)
            nlp_model: Pre-trained NLP model (optional)
            default_alpha: Default CF weight (1-alpha = NLP weight)
        """
        super().__init__(name="HybridRecommender")
        self.cf_model = cf_model
        self.nlp_model = nlp_model
        self.default_alpha = default_alpha

        # User activity tracking (set during fit)
        self.user_activity: Optional[Dict[str, int]] = None
        self.likes_df: Optional[pd.DataFrame] = None

    def fit(
        self,
        likes_df: pd.DataFrame,
        users_df: pd.DataFrame
    ) -> 'HybridRecommender':
        """
        Train both CF and NLP models.

        Args:
            likes_df: DataFrame with likes data
            users_df: DataFrame with user data

        Returns:
            Self for method chaining
        """
        logger.info(f"Training {self.name}...")
        logger.info(f"Data: {len(users_df)} users, {len(likes_df)} likes")

        # Store likes for activity tracking
        self.likes_df = likes_df.copy()

        # Calculate user activity
        logger.info("Computing user activity levels...")
        self.user_activity = likes_df.groupby('user_id').size().to_dict()
        logger.info(f"User activity computed: {len(self.user_activity)} users with interactions")

        # Train CF model
        if self.cf_model is None:
            logger.info("\n--- Training Collaborative Filtering Model ---")
            self.cf_model = CollaborativeFilteringRecommender(
                n_similar_users=50,
                min_interactions=5
            )
        else:
            logger.info("\n--- Using Pre-trained CF Model ---")

        if not self.cf_model.is_trained:
            self.cf_model.fit(likes_df, users_df)

        # Train NLP model
        if self.nlp_model is None:
            logger.info("\n--- Training NLP Model ---")
            from src.models.nlp_recommender import NLPRecommender
            self.nlp_model = NLPRecommender(
                model_name='all-MiniLM-L6-v2',
                batch_size=64
            )
        else:
            logger.info("\n--- Using Pre-trained NLP Model ---")

        if not self.nlp_model.is_trained:
            self.nlp_model.fit(users_df)

        self.is_trained = True
        logger.info(f"\n✓ {self.name} training complete")

        return self

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        exclude_ids: Optional[Set[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Generate top-K hybrid recommendations.

        Args:
            user_id: ID of the user to generate recommendations for
            k: Number of recommendations to return
            exclude_ids: Optional set of user IDs to exclude

        Returns:
            List of (user_id, score) tuples, sorted by score descending

        Raises:
            ValueError: If model not trained
        """
        self._check_trained()

        if exclude_ids is None:
            exclude_ids = set()

        # Get adaptive weights based on user activity
        alpha_cf, alpha_nlp = self._compute_adaptive_weights(user_id)

        logger.debug(
            f"User {user_id[:8]}... weights: CF={alpha_cf:.2f}, NLP={alpha_nlp:.2f}"
        )

        # Get recommendations from both models
        # Request more candidates since we'll be combining scores
        k_fetch = min(k * 3, 100)  # Fetch 3x or max 100 candidates

        # Get CF recommendations
        cf_scores = {}
        if self.cf_model is not None:
            try:
                cf_recs = self.cf_model.recommend(user_id, k=k_fetch, exclude_ids=exclude_ids)
                cf_scores = {uid: score for uid, score in cf_recs}
            except (ValueError, KeyError) as e:
                logger.warning(f"CF model failed for user {user_id}: {e}. Using NLP only.")

        # Get NLP recommendations
        nlp_scores = {}
        if self.nlp_model is not None:
            try:
                nlp_recs = self.nlp_model.recommend(user_id, k=k_fetch, exclude_ids=exclude_ids)
                nlp_scores = {uid: score for uid, score in nlp_recs}
            except (ValueError, KeyError) as e:
                logger.warning(f"NLP model failed for user {user_id}: {e}. Using CF only.")

        # Handle edge cases
        if not cf_scores and not nlp_scores:
            logger.warning(f"Both models failed for user {user_id}. Returning empty recommendations.")
            return []

        if not cf_scores:
            logger.info(f"Using NLP-only recommendations for user {user_id}")
            return sorted(nlp_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        if not nlp_scores:
            logger.info(f"Using CF-only recommendations for user {user_id}")
            return sorted(cf_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        # Normalize scores to [0, 1] range
        cf_scores_norm = self._normalize_scores(cf_scores)
        nlp_scores_norm = self._normalize_scores(nlp_scores)

        # Combine scores from both models
        combined_scores = {}
        all_candidates = set(cf_scores_norm.keys()) | set(nlp_scores_norm.keys())

        for candidate_id in all_candidates:
            # Get scores (default to 0 if not in one model)
            cf_score = cf_scores_norm.get(candidate_id, 0.0)
            nlp_score = nlp_scores_norm.get(candidate_id, 0.0)

            # Weighted combination
            combined_score = alpha_cf * cf_score + alpha_nlp * nlp_score

            combined_scores[candidate_id] = combined_score

        # Sort and return top-K
        recommendations = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]

        logger.debug(
            f"Generated {len(recommendations)} hybrid recommendations for user {user_id}"
        )

        return recommendations

    def batch_recommend(
        self,
        user_ids: List[str],
        k: int = 10,
        exclude_ids_dict: Optional[Dict[str, Set[str]]] = None
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Generate recommendations for multiple users in batch (VECTORIZED).

        This method provides significant speedup when evaluating many users by:
        - Using batch_recommend() on underlying models when available
        - Minimizing Python overhead
        - Processing multiple users efficiently

        Args:
            user_ids: List of user IDs to generate recommendations for
            k: Number of recommendations per user
            exclude_ids_dict: Optional dict mapping user_id -> set of excluded IDs

        Returns:
            Dictionary mapping user_id -> list of (recommended_id, score) tuples
        """
        self._check_trained()

        if exclude_ids_dict is None:
            exclude_ids_dict = {}

        results = {}
        k_fetch = min(k * 3, 100)  # Fetch 3x candidates for combining

        # Check if underlying models support batch recommendations
        cf_has_batch = hasattr(self.cf_model, 'batch_recommend')
        nlp_has_batch = hasattr(self.nlp_model, 'batch_recommend')

        # Get CF recommendations (batched if available)
        cf_batch_scores = {}
        if self.cf_model is not None:
            if cf_has_batch:
                logger.debug("Using CF batch_recommend()")
                try:
                    cf_batch_results = self.cf_model.batch_recommend(
                        user_ids, k=k_fetch, exclude_ids_dict=exclude_ids_dict
                    )
                    # Convert to scores dict
                    for user_id, recs in cf_batch_results.items():
                        cf_batch_scores[user_id] = {uid: score for uid, score in recs}
                except Exception as e:
                    logger.warning(f"CF batch_recommend failed: {e}. Using sequential.")
                    cf_has_batch = False

        # Get NLP recommendations (batched if available)
        nlp_batch_scores = {}
        if self.nlp_model is not None:
            if nlp_has_batch:
                logger.debug("Using NLP batch_recommend()")
                try:
                    nlp_batch_results = self.nlp_model.batch_recommend(
                        user_ids, k=k_fetch, exclude_ids_dict=exclude_ids_dict
                    )
                    # Convert to scores dict
                    for user_id, recs in nlp_batch_results.items():
                        nlp_batch_scores[user_id] = {uid: score for uid, score in recs}
                except Exception as e:
                    logger.warning(f"NLP batch_recommend failed: {e}. Using sequential.")
                    nlp_has_batch = False

        # Process each user
        for user_id in user_ids:
            exclude_ids = exclude_ids_dict.get(user_id, set())

            # Get weights for this user
            alpha_cf, alpha_nlp = self._compute_adaptive_weights(user_id)

            # Get CF scores (either from batch or sequential)
            cf_scores = {}
            if self.cf_model is not None:
                if cf_has_batch and user_id in cf_batch_scores:
                    cf_scores = cf_batch_scores[user_id]
                else:
                    try:
                        cf_recs = self.cf_model.recommend(user_id, k=k_fetch, exclude_ids=exclude_ids)
                        cf_scores = {uid: score for uid, score in cf_recs}
                    except (ValueError, KeyError) as e:
                        logger.warning(f"CF model failed for user {user_id}: {e}")

            # Get NLP scores (either from batch or sequential)
            nlp_scores = {}
            if self.nlp_model is not None:
                if nlp_has_batch and user_id in nlp_batch_scores:
                    nlp_scores = nlp_batch_scores[user_id]
                else:
                    try:
                        nlp_recs = self.nlp_model.recommend(user_id, k=k_fetch, exclude_ids=exclude_ids)
                        nlp_scores = {uid: score for uid, score in nlp_recs}
                    except (ValueError, KeyError) as e:
                        logger.warning(f"NLP model failed for user {user_id}: {e}")

            # Handle edge cases
            if not cf_scores and not nlp_scores:
                results[user_id] = []
                continue

            if not cf_scores:
                results[user_id] = sorted(nlp_scores.items(), key=lambda x: x[1], reverse=True)[:k]
                continue

            if not nlp_scores:
                results[user_id] = sorted(cf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
                continue

            # Normalize and combine scores
            cf_scores_norm = self._normalize_scores(cf_scores)
            nlp_scores_norm = self._normalize_scores(nlp_scores)

            combined_scores = {}
            all_candidates = set(cf_scores_norm.keys()) | set(nlp_scores_norm.keys())

            for candidate_id in all_candidates:
                cf_score = cf_scores_norm.get(candidate_id, 0.0)
                nlp_score = nlp_scores_norm.get(candidate_id, 0.0)
                combined_scores[candidate_id] = alpha_cf * cf_score + alpha_nlp * nlp_score

            # Sort and store top-K
            recommendations = sorted(
                combined_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:k]

            results[user_id] = recommendations

        logger.info(f"Batch recommendation complete: {len(user_ids)} users")
        return results

    def score(
        self,
        user_id: str,
        candidate_ids: List[str]
    ) -> Dict[str, float]:
        """
        Compute hybrid scores for specific candidates.

        Args:
            user_id: ID of the user
            candidate_ids: List of candidate user IDs to score

        Returns:
            Dictionary mapping candidate_id to hybrid score
        """
        self._check_trained()

        # Get adaptive weights
        alpha_cf, alpha_nlp = self._compute_adaptive_weights(user_id)

        # Get scores from both models
        cf_scores = {}
        if self.cf_model is not None:
            try:
                cf_scores = self.cf_model.score(user_id, candidate_ids)
            except (ValueError, KeyError):
                cf_scores = {cid: 0.0 for cid in candidate_ids}
        else:
            cf_scores = {cid: 0.0 for cid in candidate_ids}

        nlp_scores = {}
        if self.nlp_model is not None:
            try:
                nlp_scores = self.nlp_model.score(user_id, candidate_ids)
            except (ValueError, KeyError):
                nlp_scores = {cid: 0.0 for cid in candidate_ids}
        else:
            nlp_scores = {cid: 0.0 for cid in candidate_ids}

        # Normalize scores
        cf_scores_norm = self._normalize_scores(cf_scores)
        nlp_scores_norm = self._normalize_scores(nlp_scores)

        # Combine scores
        hybrid_scores = {}
        for candidate_id in candidate_ids:
            cf_score = cf_scores_norm.get(candidate_id, 0.0)
            nlp_score = nlp_scores_norm.get(candidate_id, 0.0)

            hybrid_scores[candidate_id] = alpha_cf * cf_score + alpha_nlp * nlp_score

        return hybrid_scores

    def save(self, filepath: Path) -> None:
        """
        Save the hybrid model to disk.

        Saves both CF and NLP models plus hybrid configuration.

        Args:
            filepath: Base path to save the model (without extension)
        """
        self._check_trained()

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save CF model
        cf_path = filepath.parent / f"{filepath.stem}_cf.pkl"
        self.cf_model.save(cf_path)
        logger.info(f"CF model saved to {cf_path}")

        # Save NLP model
        nlp_path = filepath.parent / f"{filepath.stem}_nlp"
        self.nlp_model.save(nlp_path)
        logger.info(f"NLP model saved to {nlp_path}")

        # Save hybrid configuration
        hybrid_config = {
            'name': self.name,
            'default_alpha': self.default_alpha,
            'cf_model_path': str(cf_path),
            'nlp_model_path': str(nlp_path),
            'user_activity': self.user_activity,
            'is_trained': self.is_trained
        }

        config_path = filepath.with_suffix('.pkl')
        with open(config_path, 'wb') as f:
            pickle.dump(hybrid_config, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"Hybrid config saved to {config_path}")
        logger.info(f"✓ Hybrid model saved successfully")

    def load(self, filepath: Path) -> 'HybridRecommender':
        """
        Load a trained hybrid model from disk.

        Args:
            filepath: Base path to load the model from (without extension)

        Returns:
            Self for method chaining
        """
        filepath = Path(filepath)
        config_path = filepath.with_suffix('.pkl')

        if not config_path.exists():
            raise FileNotFoundError(f"Hybrid config file not found: {config_path}")

        # Load configuration
        with open(config_path, 'rb') as f:
            config = pickle.load(f)

        self.name = config['name']
        self.default_alpha = config['default_alpha']
        self.user_activity = config['user_activity']
        self.is_trained = config['is_trained']

        # Load CF model
        cf_path = Path(config['cf_model_path'])
        logger.info(f"Loading CF model from {cf_path}...")
        self.cf_model = CollaborativeFilteringRecommender()
        self.cf_model.load(cf_path)

        # Load NLP model
        nlp_path = Path(config['nlp_model_path'])
        logger.info(f"Loading NLP model from {nlp_path}...")
        from src.models.nlp_recommender import NLPRecommender
        self.nlp_model = NLPRecommender()
        self.nlp_model.load(nlp_path)

        logger.info(f"✓ Hybrid model loaded from {filepath}")

        return self

    def _get_user_activity_level(self, user_id: str) -> int:
        """
        Get number of likes user has made.

        Args:
            user_id: User ID

        Returns:
            Number of likes (0 if user not found)
        """
        return self.user_activity.get(user_id, 0)

    def _compute_adaptive_weights(self, user_id: str) -> Tuple[float, float]:
        """
        Compute adaptive weights based on user activity.

        Args:
            user_id: User ID

        Returns:
            Tuple of (alpha_cf, alpha_nlp) weights
        """
        activity_level = self._get_user_activity_level(user_id)

        # Adaptive weighting based on activity
        if activity_level < 10:
            # New user - rely more on content/NLP
            alpha_cf, alpha_nlp = 0.3, 0.7
            level = "new"
        elif activity_level < 50:
            # Moderate user - balanced approach
            alpha_cf, alpha_nlp = 0.5, 0.5
            level = "moderate"
        else:
            # Active user - rely more on collaborative filtering
            alpha_cf, alpha_nlp = 0.7, 0.3
            level = "active"

        logger.debug(
            f"User {user_id[:8]}... activity: {activity_level} likes "
            f"({level}) -> weights: CF={alpha_cf}, NLP={alpha_nlp}"
        )

        return alpha_cf, alpha_nlp

    def _normalize_scores(self, scores_dict: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize scores to [0, 1] range using min-max scaling.

        Args:
            scores_dict: Dictionary of user_id -> score

        Returns:
            Dictionary with normalized scores
        """
        if not scores_dict:
            return {}

        scores = list(scores_dict.values())
        min_score = min(scores)
        max_score = max(scores)

        # Avoid division by zero
        if max_score == min_score:
            # All scores are the same - return equal normalized scores
            return {uid: 1.0 for uid in scores_dict.keys()}

        # Min-max normalization
        normalized = {}
        for user_id, score in scores_dict.items():
            normalized[user_id] = (score - min_score) / (max_score - min_score)

        return normalized

    def get_model_contributions(
        self,
        user_id: str,
        candidate_id: str
    ) -> Dict[str, float]:
        """
        Get individual model contributions for a candidate.

        Useful for debugging and understanding recommendations.

        Args:
            user_id: Target user ID
            candidate_id: Candidate user ID

        Returns:
            Dictionary with CF score, NLP score, weights, and final score
        """
        self._check_trained()

        # Get weights
        alpha_cf, alpha_nlp = self._compute_adaptive_weights(user_id)

        # Get raw scores
        cf_score = 0.0
        if self.cf_model is not None:
            try:
                cf_score = self.cf_model.score(user_id, [candidate_id]).get(candidate_id, 0.0)
            except (ValueError, KeyError):
                cf_score = 0.0

        nlp_score = 0.0
        if self.nlp_model is not None:
            try:
                nlp_score = self.nlp_model.score(user_id, [candidate_id]).get(candidate_id, 0.0)
            except (ValueError, KeyError):
                nlp_score = 0.0

        # Normalize (in context of just this candidate)
        cf_norm = cf_score  # Keep raw for inspection
        nlp_norm = nlp_score

        # Final score
        final_score = alpha_cf * cf_norm + alpha_nlp * nlp_norm

        return {
            'cf_score_raw': cf_score,
            'nlp_score_raw': nlp_score,
            'cf_weight': alpha_cf,
            'nlp_weight': alpha_nlp,
            'final_score': final_score,
            'user_activity': self._get_user_activity_level(user_id)
        }

    def explain_recommendation(
        self,
        user_id: str,
        candidate_id: str
    ) -> str:
        """
        Generate human-readable explanation for a recommendation.

        Args:
            user_id: Target user ID
            candidate_id: Candidate user ID

        Returns:
            Explanation string
        """
        contrib = self.get_model_contributions(user_id, candidate_id)

        explanation = f"""
Recommendation Explanation for User {user_id[:8]}... -> Candidate {candidate_id[:8]}...

User Activity: {contrib['user_activity']} likes

Model Contributions:
  • Collaborative Filtering: {contrib['cf_score_raw']:.4f} (weight: {contrib['cf_weight']:.1%})
  • NLP Semantic Match: {contrib['nlp_score_raw']:.4f} (weight: {contrib['nlp_weight']:.1%})

Final Hybrid Score: {contrib['final_score']:.4f}

Strategy: {'CF-heavy (active user)' if contrib['cf_weight'] > 0.6 else 'NLP-heavy (new user)' if contrib['nlp_weight'] > 0.6 else 'Balanced approach'}
"""
        return explanation.strip()

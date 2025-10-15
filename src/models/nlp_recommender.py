"""
NLP-based semantic similarity recommender using Sentence-BERT.

This module implements a recommendation system based on profile text similarity
using pre-trained sentence transformer models.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Set
from pathlib import Path
import pickle
from tqdm import tqdm

from sentence_transformers import SentenceTransformer

from src.models.base_recommender import BaseRecommender
from src.preprocessing import DataPreprocessor
from src.utils import setup_logger

logger = setup_logger(__name__)


class NLPRecommender(BaseRecommender):
    """
    Semantic similarity recommender using Sentence-BERT.

    Uses pre-trained sentence transformers to encode user profiles and
    recommends based on semantic similarity of profile text.
    """

    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        batch_size: int = 32,
        device: Optional[str] = None
    ):
        """
        Initialize NLP recommender.

        Args:
            model_name: Sentence-transformer model name
            batch_size: Batch size for embedding generation
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None.
        """
        super().__init__(name="NLPRecommender")
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device

        # Model components (set during fit)
        self.model: Optional[SentenceTransformer] = None
        self.embeddings: Optional[np.ndarray] = None
        self.similarity_matrix: Optional[np.ndarray] = None
        self.user_id_to_idx: Optional[Dict[str, int]] = None
        self.idx_to_user_id: Optional[Dict[int, str]] = None
        self.users_df: Optional[pd.DataFrame] = None
        self.profile_texts: Optional[pd.Series] = None
        self.preprocessor: Optional[DataPreprocessor] = None

    def fit(self, users_df: pd.DataFrame) -> 'NLPRecommender':
        """
        Train the NLP recommender by generating embeddings.

        Args:
            users_df: DataFrame with user data

        Returns:
            Self for method chaining

        Raises:
            ValueError: If required columns are missing
        """
        logger.info(f"Training {self.name}...")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Users: {len(users_df)}")

        # Validate required columns
        required_cols = ['user_id', 'gender', 'age', 'city']
        missing_cols = [col for col in required_cols if col not in users_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        if len(users_df) == 0:
            raise ValueError("users_df cannot be empty")

        self.preprocessor = DataPreprocessor()
        self.users_df = users_df.copy()

        # Load sentence transformer model
        logger.info(f"Loading sentence transformer model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        logger.info(f"Model loaded on device: {self.model.device}")

        # Create user mappings
        all_user_ids = users_df['user_id'].values
        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(all_user_ids)}
        self.idx_to_user_id = {idx: uid for uid, idx in self.user_id_to_idx.items()}

        # Generate profile texts
        logger.info("Generating profile texts...")
        self.profile_texts = self.preprocessor.create_profile_text(users_df)

        # Check for empty bios
        empty_bio_count = (users_df['bio'].isna() | (users_df['bio'] == '')).sum()
        if empty_bio_count > 0:
            logger.info(f"Note: {empty_bio_count} users have empty bios (using job/school/city only)")

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(self.profile_texts)} users...")
        logger.info(f"Batch size: {self.batch_size}")

        # CRITICAL: Ensure texts are in same order as user_id_to_idx mapping
        # profile_texts is a Series indexed by user_id, we need to preserve order
        texts = [self.profile_texts.loc[uid] for uid in all_user_ids]

        # Encode with progress bar
        # Note: Most sentence-transformers have max length of 256-512 tokens
        # The model will automatically truncate, but we set it explicitly
        self.embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalization for cosine similarity
            convert_to_tensor=False  # Ensure numpy output
        )

        logger.info(f"Embeddings shape: {self.embeddings.shape}")
        logger.info(f"Embedding dimension: {self.embeddings.shape[1]}")

        # Pre-compute similarity matrix for fast inference
        logger.info("Pre-computing similarity matrix...")
        from sklearn.metrics.pairwise import cosine_similarity

        self.similarity_matrix = cosine_similarity(self.embeddings)
        logger.info(f"Similarity matrix shape: {self.similarity_matrix.shape}")

        # Log memory usage
        matrix_mb = self.similarity_matrix.nbytes / (1024 * 1024)
        logger.info(f"Similarity matrix memory: {matrix_mb:.2f} MB")

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
        Generate top-K recommendations based on semantic similarity.

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

        if exclude_ids is None:
            exclude_ids = set()

        # Add target user to exclusions
        exclude_ids = exclude_ids | {user_id}

        # Get user index
        user_idx = self.user_id_to_idx[user_id]

        # Get pre-computed similarities for this user
        similarities = self.similarity_matrix[user_idx]

        # Get user for preference filtering
        user = self.users_df[self.users_df['user_id'] == user_id].iloc[0]
        user_gender = user['gender']
        user_pref_gender = user.get('matching_pref_gender')
        user_pref_age_min = user.get('matching_pref_age_min')
        user_pref_age_max = user.get('matching_pref_age_max')
        user_pref_use_location = user.get('matching_pref_use_location', False)
        user_city = user.get('city')

        # Create boolean mask for valid candidates (vectorized - FAST!)
        valid_mask = np.ones(len(similarities), dtype=bool)

        # Exclude target user
        valid_mask[user_idx] = False

        # Exclude other users if specified (vectorized - no loop!)
        if exclude_ids:
            exclude_indices = [self.user_id_to_idx[uid] for uid in exclude_ids if uid in self.user_id_to_idx]
            if exclude_indices:
                valid_mask[exclude_indices] = False

        # Apply gender filter (vectorized)
        candidate_genders = self.users_df['gender'].values
        if pd.notna(user_pref_gender) and user_pref_gender != 'both':
            valid_mask &= (candidate_genders == user_pref_gender)

        # Apply age filter (vectorized)
        if pd.notna(user_pref_age_min) and pd.notna(user_pref_age_max):
            candidate_ages = self.users_df['age'].values
            valid_mask &= (candidate_ages >= user_pref_age_min) & (candidate_ages <= user_pref_age_max)

        # Apply location filter (vectorized)
        if user_pref_use_location and pd.notna(user_city):
            candidate_cities = self.users_df['city'].values
            valid_mask &= (candidate_cities == user_city)

        # Apply mask to similarities
        valid_similarities = similarities.copy()
        valid_similarities[~valid_mask] = -1  # Set invalid candidates to -1

        # Get top-K indices using vectorized sort (FAST!)
        top_k_indices = np.argsort(valid_similarities)[::-1][:k]

        # Build recommendations list (only iterating K times, not 20,000!)
        recommendations = []
        for idx in top_k_indices:
            if valid_similarities[idx] > 0:
                candidate_id = self.idx_to_user_id[idx]
                recommendations.append((candidate_id, float(valid_similarities[idx])))

        logger.debug(f"Generated {len(recommendations)} recommendations for user {user_id}")

        return recommendations

    def batch_recommend(
        self,
        user_ids: List[str],
        k: int = 10,
        exclude_ids_dict: Optional[Dict[str, Set[str]]] = None
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Generate recommendations for multiple users in batch.

        This method is more efficient than calling recommend() in a loop
        as it minimizes repeated operations.

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

        for user_id in user_ids:
            # Check if user exists
            if user_id not in self.user_id_to_idx:
                logger.warning(f"User {user_id} not found in training data")
                results[user_id] = []
                continue

            # Get exclude set for this user
            exclude_ids = exclude_ids_dict.get(user_id, set())

            # Generate recommendations
            try:
                recommendations = self.recommend(user_id, k=k, exclude_ids=exclude_ids)
                results[user_id] = recommendations
            except Exception as e:
                logger.warning(f"Error generating recommendations for {user_id}: {e}")
                results[user_id] = []

        logger.debug(f"Batch recommendation complete: {len(user_ids)} users")
        return results

    def score(
        self,
        user_id: str,
        candidate_ids: List[str]
    ) -> Dict[str, float]:
        """
        Compute semantic similarity scores for specific candidates.

        Args:
            user_id: ID of the user
            candidate_ids: List of candidate user IDs to score

        Returns:
            Dictionary mapping candidate_id to score
        """
        self._check_trained()

        if user_id not in self.user_id_to_idx:
            raise ValueError(f"User {user_id} not found in training data")

        # Get user index
        user_idx = self.user_id_to_idx[user_id]

        # Get pre-computed similarities for this user
        user_similarities = self.similarity_matrix[user_idx]

        # Score each candidate using pre-computed similarities
        scores = {}
        for candidate_id in candidate_ids:
            if candidate_id not in self.user_id_to_idx:
                scores[candidate_id] = 0.0
                continue

            candidate_idx = self.user_id_to_idx[candidate_id]
            scores[candidate_id] = float(user_similarities[candidate_idx])

        return scores

    def save(self, filepath: Path) -> None:
        """
        Save the trained model to disk.

        Saves embeddings as .npy file and metadata as pickle.

        Args:
            filepath: Path to save the model (without extension)
        """
        self._check_trained()

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save embeddings as numpy array
        embeddings_path = filepath.with_suffix('.npy')
        np.save(embeddings_path, self.embeddings)
        logger.info(f"Embeddings saved to {embeddings_path}")

        # Save similarity matrix
        similarity_path = filepath.parent / f"{filepath.stem}_similarity.npy"
        np.save(similarity_path, self.similarity_matrix)
        logger.info(f"Similarity matrix saved to {similarity_path}")

        # Save metadata
        metadata = {
            'name': self.name,
            'model_name': self.model_name,
            'batch_size': self.batch_size,
            'user_id_to_idx': self.user_id_to_idx,
            'idx_to_user_id': self.idx_to_user_id,
            'users_df': self.users_df,
            'profile_texts': self.profile_texts,
            'is_trained': self.is_trained,
            'embedding_shape': self.embeddings.shape,
            'similarity_matrix_path': str(similarity_path)
        }

        metadata_path = filepath.with_suffix('.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"Metadata saved to {metadata_path}")

    def load(self, filepath: Path) -> 'NLPRecommender':
        """
        Load a trained model from disk.

        Args:
            filepath: Path to load the model from (without extension)

        Returns:
            Self for method chaining
        """
        filepath = Path(filepath)

        # Load embeddings
        embeddings_path = filepath.with_suffix('.npy')
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

        self.embeddings = np.load(embeddings_path)
        logger.info(f"Embeddings loaded from {embeddings_path}")

        # Load metadata
        metadata_path = filepath.with_suffix('.pkl')
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        self.name = metadata['name']
        self.model_name = metadata['model_name']
        self.batch_size = metadata['batch_size']
        self.user_id_to_idx = metadata['user_id_to_idx']
        self.idx_to_user_id = metadata['idx_to_user_id']
        self.users_df = metadata['users_df']
        self.profile_texts = metadata['profile_texts']
        self.is_trained = metadata['is_trained']

        # Load similarity matrix
        similarity_path = Path(metadata.get('similarity_matrix_path',
                                           filepath.parent / f"{filepath.stem}_similarity.npy"))
        if similarity_path.exists():
            self.similarity_matrix = np.load(similarity_path)
            logger.info(f"Similarity matrix loaded from {similarity_path}")
        else:
            # Fallback: recompute if not found
            logger.warning(f"Similarity matrix not found at {similarity_path}, recomputing...")
            from sklearn.metrics.pairwise import cosine_similarity
            self.similarity_matrix = cosine_similarity(self.embeddings)
            logger.info(f"Similarity matrix recomputed: {self.similarity_matrix.shape}")

        # Load sentence transformer model for potential re-encoding
        logger.info(f"Loading sentence transformer model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name, device=self.device)

        logger.info(f"Model loaded from {filepath}")

        return self

    def _compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score
        """
        # Since embeddings are already normalized, dot product = cosine similarity
        return np.dot(embedding1, embedding2)

    def _compute_all_similarities(self, user_embedding: np.ndarray) -> np.ndarray:
        """
        Compute similarities with all users.

        Args:
            user_embedding: Target user's embedding

        Returns:
            Array of similarity scores
        """
        # Matrix multiplication for efficiency (embeddings are normalized)
        similarities = np.dot(self.embeddings, user_embedding)
        return similarities

    def _passes_preferences(self, user: pd.Series, candidate_id: str) -> bool:
        """
        Check if candidate passes user's preferences.

        Args:
            user: User Series
            candidate_id: Candidate user ID

        Returns:
            True if candidate passes all filters
        """
        # Get candidate
        candidate = self.users_df[self.users_df['user_id'] == candidate_id]

        if candidate.empty:
            return False

        candidate = candidate.iloc[0]

        # Gender filter
        preferred_gender = user.get('matching_pref_gender')
        if pd.notna(preferred_gender) and preferred_gender != 'both':
            if candidate['gender'] != preferred_gender:
                return False

        # Age filter
        if pd.notna(user.get('matching_pref_age_min')) and pd.notna(user.get('matching_pref_age_max')):
            if not (user['matching_pref_age_min'] <= candidate['age'] <= user['matching_pref_age_max']):
                return False

        # Location filter
        if user.get('matching_pref_use_location', False):
            if candidate['city'] != user['city']:
                return False

        return True

    def get_similar_profiles(
        self,
        user_id: str,
        k: int = 5
    ) -> List[Tuple[str, str, float]]:
        """
        Get similar user profiles with their text for inspection.

        Args:
            user_id: Target user ID
            k: Number of similar profiles to return

        Returns:
            List of (user_id, profile_text, similarity) tuples
        """
        self._check_trained()

        recommendations = self.recommend(user_id, k=k, exclude_ids=set())

        results = []
        for rec_user_id, score in recommendations:
            profile_text = self.profile_texts.loc[rec_user_id]
            results.append((rec_user_id, profile_text, score))

        return results

    def encode_new_user(self, profile_text: str) -> np.ndarray:
        """
        Encode a new user's profile text.

        Useful for real-time recommendations for new users.

        Args:
            profile_text: User's profile text

        Returns:
            Embedding vector

        Raises:
            ValueError: If profile_text is empty or None
        """
        self._check_trained()

        if not profile_text or not profile_text.strip():
            raise ValueError("profile_text cannot be empty")

        # Clean text using same preprocessing as training
        import re
        text = profile_text.lower().strip()
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        if len(text) > 800:
            text = text[:800] + "..."

        embedding = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            convert_to_tensor=False
        )

        return embedding[0]

"""
Abstract base class for recommendation models.

This module defines the interface that all recommender systems must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional, Set
from pathlib import Path


class BaseRecommender(ABC):
    """
    Abstract base class for all recommendation models.

    All recommender systems must inherit from this class
    and implement the required methods for training, recommendation, and scoring.
    """

    def __init__(self, name: str = "BaseRecommender"):
        """
        Initialize the base recommender.

        Args:
            name: Name identifier for this recommender
        """
        self.name = name
        self.is_trained = False

    @abstractmethod
    def fit(self, *args, **kwargs) -> 'BaseRecommender':
        """
        Train the recommendation model.

        Args:
            *args: Training data (model-specific)
            **kwargs: Additional training parameters

        Returns:
            Self for method chaining

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclass must implement fit()")

    @abstractmethod
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
            exclude_ids: Optional set of user IDs to exclude from recommendations
                        (e.g., users already liked/matched)

        Returns:
            List of (user_id, score) tuples, sorted by score descending

        Raises:
            ValueError: If user_id is invalid or model not trained
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclass must implement recommend()")

    @abstractmethod
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

        Raises:
            ValueError: If user_id is invalid or model not trained
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclass must implement score()")

    @abstractmethod
    def save(self, filepath: Path) -> None:
        """
        Save the trained model to disk.

        Args:
            filepath: Path to save the model

        Raises:
            ValueError: If model not trained
            IOError: If save fails
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclass must implement save()")

    @abstractmethod
    def load(self, filepath: Path) -> 'BaseRecommender':
        """
        Load a trained model from disk.

        Args:
            filepath: Path to load the model from

        Returns:
            Self for method chaining

        Raises:
            FileNotFoundError: If file doesn't exist
            IOError: If load fails
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclass must implement load()")

    def _check_trained(self) -> None:
        """
        Check if model has been trained.

        Raises:
            ValueError: If model has not been trained
        """
        if not self.is_trained:
            raise ValueError(
                f"{self.name} must be trained before making recommendations. "
                "Call fit() first."
            )

    def __repr__(self) -> str:
        """String representation of the recommender."""
        trained_status = "trained" if self.is_trained else "not trained"
        return f"{self.name}({trained_status})"

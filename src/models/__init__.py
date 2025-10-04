"""
Recommendation models for WorkHeart dating app.

This package contains various recommendation algorithms including:
- Collaborative filtering (user-user similarity)
- NLP-based semantic matching (Sentence-BERT)
- Hybrid approaches combining multiple signals
"""

# Lazy imports to avoid loading heavy dependencies unless needed
__all__ = [
    'BaseRecommender',
    'CollaborativeFilteringRecommender',
    'NLPRecommender',
    'HybridRecommender',
]


def __getattr__(name):
    """Lazy import of models to avoid loading sentence-transformers unless needed."""
    if name == 'BaseRecommender':
        from src.models.base_recommender import BaseRecommender
        return BaseRecommender
    elif name == 'CollaborativeFilteringRecommender':
        from src.models.cf_recommender import CollaborativeFilteringRecommender
        return CollaborativeFilteringRecommender
    elif name == 'NLPRecommender':
        from src.models.nlp_recommender import NLPRecommender
        return NLPRecommender
    elif name == 'HybridRecommender':
        from src.models.hybrid_recommender import HybridRecommender
        return HybridRecommender
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

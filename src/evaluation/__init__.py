"""
Evaluation module for recommendation systems.

Provides tools for evaluating model performance, including:
- Standard metrics (Precision, Recall, NDCG)
- Segmented evaluation (cold/warm/active users)
- Batch evaluation with stratified sampling
- Performance analysis and reporting
"""

from src.evaluation.cold_start_evaluator import (
    ColdStartEvaluator,
    evaluate_with_segments
)

from src.evaluation.batch_evaluator import (
    BatchEvaluator,
    evaluate_with_sampling
)

__all__ = [
    'ColdStartEvaluator',
    'evaluate_with_segments',
    'BatchEvaluator',
    'evaluate_with_sampling'
]

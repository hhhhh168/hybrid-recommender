"""
Comprehensive evaluation system for recommendation models.

This module provides metrics and evaluation tools for comparing CF, NLP, and Hybrid
recommender systems on various performance dimensions including accuracy, coverage,
and diversity.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path
import json
import warnings
from tqdm import tqdm

from src.models.base_recommender import BaseRecommender
from src.utils import setup_logger

logger = setup_logger(__name__)


class RecommenderEvaluator:
    """
    Evaluates and compares recommendation models.

    Computes standard recommendation metrics including precision, recall, NDCG,
    MAP, coverage, and diversity. Generates comparison reports and visualizations.
    """

    def __init__(
        self,
        models_dict: Dict[str, BaseRecommender],
        test_data: pd.DataFrame,
        users_df: pd.DataFrame,
        train_data: Optional[pd.DataFrame] = None
    ):
        """
        Initialize evaluator.

        Args:
            models_dict: Dict mapping model names to trained models
                        e.g., {'CF': cf_model, 'NLP': nlp_model, 'Hybrid': hybrid_model}
            test_data: DataFrame with test interactions (user_id, liked_user_id, action)
            users_df: DataFrame with user data for context
            train_data: DataFrame with training interactions (for proper exclusion)

        Raises:
            ValueError: If models_dict is empty or test_data is invalid
        """
        if not models_dict:
            raise ValueError("models_dict cannot be empty")

        if test_data.empty:
            raise ValueError("test_data cannot be empty")

        required_cols = ['user_id', 'liked_user_id']
        missing_cols = [col for col in required_cols if col not in test_data.columns]
        if missing_cols:
            raise ValueError(f"test_data missing required columns: {missing_cols}")

        self.models = models_dict
        self.test_data = test_data.copy()
        self.train_data = train_data.copy() if train_data is not None else None
        self.users_df = users_df.copy()

        # Precompute ground truth for all users
        logger.info("Precomputing ground truth from test data...")
        self.ground_truth = self._build_ground_truth()
        logger.info(f"Ground truth computed for {len(self.ground_truth)} users")

        # Precompute training interactions per user (CRITICAL FIX!)
        if self.train_data is not None:
            logger.info("Precomputing training interactions for proper exclusion...")
            self.train_interactions = self._build_train_interactions()
            logger.info(f"Training interactions computed for {len(self.train_interactions)} users")
        else:
            logger.warning("No training data provided - recommendations may include training items!")
            self.train_interactions = {}

        # Cache for user recommendations
        self._recommendation_cache: Dict[str, Dict[str, List[Tuple[str, float]]]] = {}

        # All items in the catalog
        self.all_items = set(users_df['user_id'].unique())

    def _build_ground_truth(self) -> Dict[str, Set[str]]:
        """
        Build ground truth mapping from test data.

        Returns:
            Dict mapping user_id to set of liked user_ids in test set
        """
        ground_truth = {}

        for user_id, group in self.test_data.groupby('user_id'):
            # Ground truth = users they liked in test set
            liked_users = set(group['liked_user_id'].unique())
            ground_truth[user_id] = liked_users

        return ground_truth

    def _build_train_interactions(self) -> Dict[str, Set[str]]:
        """
        Build training interactions mapping for proper exclusion.

        Returns:
            Dict mapping user_id to set of liked user_ids in training set
        """
        train_interactions = {}

        for user_id, group in self.train_data.groupby('user_id'):
            # Training interactions = users they liked in training set
            liked_users = set(group['liked_user_id'].unique())
            train_interactions[user_id] = liked_users

        return train_interactions

    def precision_at_k(
        self,
        recommendations: List[str],
        ground_truth: Set[str],
        k: int
    ) -> float:
        """
        Compute Precision@K.

        Precision@K = (# relevant items in top-K) / K

        Args:
            recommendations: List of recommended user IDs (ordered)
            ground_truth: Set of relevant user IDs
            k: Cutoff position

        Returns:
            Precision@K score (0-1)
        """
        if k <= 0:
            return 0.0

        # Take top-K recommendations
        top_k = recommendations[:k]

        # Count relevant items in top-K
        relevant_count = sum(1 for rec in top_k if rec in ground_truth)

        return relevant_count / k

    def recall_at_k(
        self,
        recommendations: List[str],
        ground_truth: Set[str],
        k: int
    ) -> float:
        """
        Compute Recall@K.

        Recall@K = (# relevant items in top-K) / (total # relevant items)

        Args:
            recommendations: List of recommended user IDs (ordered)
            ground_truth: Set of relevant user IDs
            k: Cutoff position

        Returns:
            Recall@K score (0-1)
        """
        if not ground_truth or k <= 0:
            return 0.0

        # Take top-K recommendations
        top_k = recommendations[:k]

        # Count relevant items in top-K
        relevant_count = sum(1 for rec in top_k if rec in ground_truth)

        return relevant_count / len(ground_truth)

    def ndcg_at_k(
        self,
        recommendations: List[str],
        ground_truth: Set[str],
        k: int
    ) -> float:
        """
        Compute Normalized Discounted Cumulative Gain@K.

        NDCG accounts for position of relevant items - items ranked higher
        contribute more to the score.

        DCG@K = sum(rel_i / log2(i + 1)) for i in 1..K
        IDCG@K = optimal DCG (all relevant items ranked first)
        NDCG@K = DCG@K / IDCG@K

        Args:
            recommendations: List of recommended user IDs (ordered)
            ground_truth: Set of relevant user IDs
            k: Cutoff position

        Returns:
            NDCG@K score (0-1)
        """
        if not ground_truth or k <= 0:
            return 0.0

        # Take top-K recommendations
        top_k = recommendations[:k]

        # Compute DCG@K
        dcg = 0.0
        for i, rec in enumerate(top_k):
            if rec in ground_truth:
                # Relevance = 1 for binary relevance
                # Position discount: log2(i + 2) because i is 0-indexed
                dcg += 1.0 / np.log2(i + 2)

        # Compute IDCG@K (ideal DCG if all relevant items were ranked first)
        idcg = 0.0
        num_relevant = min(len(ground_truth), k)
        for i in range(num_relevant):
            idcg += 1.0 / np.log2(i + 2)

        # Normalize
        if idcg == 0.0:
            return 0.0

        return dcg / idcg

    def mean_average_precision(
        self,
        recommendations: List[str],
        ground_truth: Set[str]
    ) -> float:
        """
        Compute Average Precision for a single user.

        AP = (sum of precision@k for each relevant item k) / (# relevant items)

        Args:
            recommendations: List of recommended user IDs (ordered)
            ground_truth: Set of relevant user IDs

        Returns:
            Average Precision score (0-1)
        """
        if not ground_truth:
            return 0.0

        relevant_count = 0
        precision_sum = 0.0

        for i, rec in enumerate(recommendations):
            if rec in ground_truth:
                relevant_count += 1
                # Precision at this position
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i

        if relevant_count == 0:
            return 0.0

        return precision_sum / len(ground_truth)

    def coverage(
        self,
        all_recommendations: List[List[str]],
        total_items: int
    ) -> float:
        """
        Compute catalog coverage.

        Coverage = (# unique items recommended) / (total # items)

        Higher coverage means recommendations are more diverse across the catalog.

        Args:
            all_recommendations: List of recommendation lists for all users
            total_items: Total number of items in catalog

        Returns:
            Coverage score (0-1)
        """
        if total_items == 0:
            return 0.0

        # Collect all unique items recommended
        recommended_items = set()
        for recs in all_recommendations:
            recommended_items.update(recs)

        return len(recommended_items) / total_items

    def diversity(self, recommendations: List[str]) -> float:
        """
        Compute intra-list diversity for a single user's recommendations.

        Diversity = 1 - average pairwise similarity

        Uses user embeddings or features to compute similarity. For now, we use
        a simple heuristic based on user attributes.

        Args:
            recommendations: List of recommended user IDs

        Returns:
            Diversity score (0-1)
        """
        if len(recommendations) < 2:
            return 1.0  # Trivially diverse

        # Get user data for recommendations
        rec_users = self.users_df[self.users_df['user_id'].isin(recommendations)]

        if rec_users.empty or len(rec_users) < 2:
            return 0.0

        # Compute pairwise similarities using simple features
        similarities = []

        for i in range(len(rec_users)):
            for j in range(i + 1, len(rec_users)):
                user_i = rec_users.iloc[i]
                user_j = rec_users.iloc[j]

                sim = self._compute_user_similarity(user_i, user_j)
                similarities.append(sim)

        if not similarities:
            return 1.0

        avg_similarity = np.mean(similarities)
        diversity_score = 1.0 - avg_similarity

        return max(0.0, min(1.0, diversity_score))

    def _compute_user_similarity(self, user1: pd.Series, user2: pd.Series) -> float:
        """
        Compute similarity between two users based on attributes.

        Args:
            user1: First user Series
            user2: Second user Series

        Returns:
            Similarity score (0-1)
        """
        similarity = 0.0
        factors = 0

        # Gender similarity (0 or 1)
        if user1['gender'] == user2['gender']:
            similarity += 1.0
        factors += 1

        # Age similarity (normalized difference)
        age_diff = abs(user1['age'] - user2['age'])
        age_sim = max(0.0, 1.0 - age_diff / 50.0)  # Normalize by 50 years
        similarity += age_sim
        factors += 1

        # Location similarity (0 or 1)
        if user1['city'] == user2['city']:
            similarity += 1.0
        factors += 1

        return similarity / factors if factors > 0 else 0.0

    def evaluate_model(
        self,
        model: BaseRecommender,
        model_name: str,
        k_values: List[int] = [5, 10, 20],
        max_users: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate a single model on all metrics using VECTORIZED batch recommendation.

        Args:
            model: Trained recommender model
            model_name: Name of the model (for caching)
            k_values: List of K values for metrics
            max_users: Maximum number of users to evaluate (for speed)

        Returns:
            Dictionary with metric names and values
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating model: {model_name}")
        logger.info(f"{'='*60}")

        metrics = {}

        # Get users to evaluate
        test_users = list(self.ground_truth.keys())

        # Sample if needed
        if max_users and len(test_users) > max_users:
            logger.info(f"Sampling {max_users} users from {len(test_users)} total")
            test_users = np.random.choice(test_users, size=max_users, replace=False).tolist()

        logger.info(f"Evaluating on {len(test_users)} users")

        # Track metrics across all users
        precision_scores = {k: [] for k in k_values}
        recall_scores = {k: [] for k in k_values}
        ndcg_scores = {k: [] for k in k_values}
        map_scores = []
        diversity_scores = []

        all_recommendations = []

        # Track errors
        error_count = 0
        cold_start_count = 0

        max_k = max(k_values)

        # Build exclude_ids dict for all users (VECTORIZED PREPROCESSING)
        logger.info("Building exclusion sets for all users...")
        exclude_ids_dict = {}
        for user_id in test_users:
            train_likes = self.train_interactions.get(user_id, set())
            exclude_ids_dict[user_id] = train_likes | {user_id}  # Exclude training items + self

        # VECTORIZED BATCH RECOMMENDATION - Single call for all users!
        logger.info(f"Generating batch recommendations for {len(test_users)} users...")

        # Check if model supports batch_recommend
        if hasattr(model, 'batch_recommend'):
            logger.info(f"Using vectorized batch_recommend() - FAST!")
            try:
                batch_results = model.batch_recommend(
                    user_ids=test_users,
                    k=max_k,
                    exclude_ids_dict=exclude_ids_dict
                )
            except Exception as e:
                logger.warning(f"Batch recommendation failed: {e}. Falling back to sequential.")
                batch_results = None
        else:
            logger.warning(f"Model {model_name} doesn't support batch_recommend(). Using sequential fallback.")
            batch_results = None

        # Fallback to sequential if batch failed
        if batch_results is None:
            logger.info("Using sequential recommend() - slower fallback...")
            batch_results = {}
            for user_id in tqdm(test_users, desc=f"Sequential {model_name}"):
                try:
                    exclude_ids = exclude_ids_dict[user_id]
                    recs = model.recommend(user_id, k=max_k, exclude_ids=exclude_ids)
                    batch_results[user_id] = recs
                except Exception as e:
                    logger.debug(f"Model failed for user {user_id}: {e}")
                    batch_results[user_id] = []

        # VECTORIZED METRIC COMPUTATION - Process all results
        logger.info("Computing metrics for all users...")
        for user_id in tqdm(test_users, desc=f"Computing metrics for {model_name}"):
            # Get ground truth
            ground_truth = self.ground_truth.get(user_id, set())

            if not ground_truth:
                logger.debug(f"User {user_id} has no ground truth, skipping")
                continue

            # Get recommendations from batch results
            recs = batch_results.get(user_id, [])

            if not recs:
                cold_start_count += 1
                logger.debug(f"No recommendations for user {user_id}")
                continue

            # Extract user IDs (recs are tuples of (user_id, score))
            rec_ids = [rec[0] for rec in recs]

            # Compute metrics at each K
            for k in k_values:
                prec = self.precision_at_k(rec_ids, ground_truth, k)
                rec = self.recall_at_k(rec_ids, ground_truth, k)
                ndcg = self.ndcg_at_k(rec_ids, ground_truth, k)

                precision_scores[k].append(prec)
                recall_scores[k].append(rec)
                ndcg_scores[k].append(ndcg)

            # Compute MAP (uses all recommendations)
            map_score = self.mean_average_precision(rec_ids, ground_truth)
            map_scores.append(map_score)

            # Compute diversity
            div = self.diversity(rec_ids)
            diversity_scores.append(div)

            # Store for coverage computation
            all_recommendations.append(rec_ids)

        # Aggregate metrics
        for k in k_values:
            if precision_scores[k]:
                metrics[f'Precision@{k}'] = np.mean(precision_scores[k])
                metrics[f'Recall@{k}'] = np.mean(recall_scores[k])
                metrics[f'NDCG@{k}'] = np.mean(ndcg_scores[k])
            else:
                metrics[f'Precision@{k}'] = 0.0
                metrics[f'Recall@{k}'] = 0.0
                metrics[f'NDCG@{k}'] = 0.0

        if map_scores:
            metrics['MAP'] = np.mean(map_scores)
        else:
            metrics['MAP'] = 0.0

        if diversity_scores:
            metrics['Diversity'] = np.mean(diversity_scores)
        else:
            metrics['Diversity'] = 0.0

        # Compute coverage
        if all_recommendations:
            metrics['Coverage'] = self.coverage(all_recommendations, len(self.all_items))
        else:
            metrics['Coverage'] = 0.0

        # Log summary
        logger.info(f"\n{model_name} Results:")
        logger.info(f"  Users evaluated: {len(all_recommendations)}")
        logger.info(f"  Errors: {error_count}")
        logger.info(f"  Cold start: {cold_start_count}")

        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")

        return metrics

    def evaluate_all_models(
        self,
        k_values: List[int] = [5, 10, 20],
        max_users: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Evaluate all models and return comparison table.

        Args:
            k_values: List of K values for metrics
            max_users: Maximum number of users to evaluate

        Returns:
            DataFrame with models as columns and metrics as rows
        """
        logger.info("\n" + "="*60)
        logger.info("EVALUATING ALL MODELS")
        logger.info("="*60)

        all_metrics = {}

        for model_name, model in self.models.items():
            metrics = self.evaluate_model(model, model_name, k_values, max_users)
            all_metrics[model_name] = metrics

        # Convert to DataFrame
        df = pd.DataFrame(all_metrics)

        # Reorder rows for better readability
        metric_order = []
        for k in k_values:
            metric_order.extend([f'Precision@{k}', f'Recall@{k}', f'NDCG@{k}'])
        metric_order.extend(['MAP', 'Coverage', 'Diversity'])

        # Reindex with available metrics
        available_metrics = [m for m in metric_order if m in df.index]
        df = df.reindex(available_metrics)

        return df

    def generate_report(
        self,
        output_path: str = 'results/metrics/',
        k_values: List[int] = [5, 10, 20],
        max_users: Optional[int] = None
    ) -> None:
        """
        Generate comprehensive evaluation report.

        Creates JSON and CSV files with metrics, plus visualizations.

        Args:
            output_path: Directory to save results
            k_values: List of K values for metrics
            max_users: Maximum number of users to evaluate
        """
        logger.info("\n" + "="*60)
        logger.info("GENERATING EVALUATION REPORT")
        logger.info("="*60)

        # Create output directory
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Evaluate all models
        metrics_df = self.evaluate_all_models(k_values, max_users)

        # Save as CSV
        csv_path = output_dir / 'comparison.csv'
        metrics_df.to_csv(csv_path)
        logger.info(f"\n✓ Comparison table saved to {csv_path}")

        # Save as JSON
        json_path = output_dir / 'evaluation_results.json'
        metrics_dict = metrics_df.to_dict()
        with open(json_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        logger.info(f"✓ Results saved to {json_path}")

        # Print summary
        logger.info("\n" + "="*60)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*60)
        logger.info("\n" + metrics_df.to_string())

        # Generate visualizations
        try:
            viz_path = output_dir / 'metrics_comparison.png'
            self.plot_metrics_comparison(metrics_df, viz_path)
            logger.info(f"\n✓ Visualization saved to {viz_path}")
        except ImportError:
            logger.warning("\nmatplotlib not available, skipping visualization")
        except Exception as e:
            logger.warning(f"\nVisualization failed: {e}")

        logger.info("\n" + "="*60)
        logger.info("REPORT GENERATION COMPLETE")
        logger.info("="*60)

    def plot_metrics_comparison(
        self,
        metrics_df: pd.DataFrame,
        save_path: Path
    ) -> None:
        """
        Create bar charts comparing models across metrics.

        Args:
            metrics_df: DataFrame with metrics (from evaluate_all_models)
            save_path: Path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for visualization")

        # Group metrics by type
        metric_groups = {
            'Precision': [col for col in metrics_df.index if col.startswith('Precision@')],
            'Recall': [col for col in metrics_df.index if col.startswith('Recall@')],
            'NDCG': [col for col in metrics_df.index if col.startswith('NDCG@')],
            'Other': ['MAP', 'Coverage', 'Diversity']
        }

        # Filter out empty groups
        metric_groups = {k: v for k, v in metric_groups.items() if v and all(m in metrics_df.index for m in v)}

        n_groups = len(metric_groups)

        if n_groups == 0:
            logger.warning("No metrics to plot")
            return

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

        for idx, (group_name, metrics) in enumerate(metric_groups.items()):
            if idx >= len(axes):
                break

            ax = axes[idx]

            # Get data for this group
            data = metrics_df.loc[metrics]

            # Plot
            x = np.arange(len(metrics))
            width = 0.8 / len(data.columns)

            for i, model_name in enumerate(data.columns):
                values = data[model_name].values
                offset = (i - len(data.columns) / 2) * width + width / 2
                ax.bar(x + offset, values, width, label=model_name, color=colors[i % len(colors)])

            ax.set_xlabel('Metric')
            ax.set_ylabel('Score')
            ax.set_title(f'{group_name} Metrics')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics, rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, max(1.0, data.max().max() * 1.1))

        # Hide unused subplots
        for idx in range(n_groups, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Metrics comparison plot saved to {save_path}")

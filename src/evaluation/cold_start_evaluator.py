"""
Cold-Start Evaluation Module

Segments users by interaction count (cold/warm/active) and evaluates
recommendation models separately on each segment to understand performance
across different user activity levels.

KEY INSIGHTS:
- Cold-start users (< 5 interactions): CF struggles, NLP should win
- Warm-start users (5-50 interactions): Hybrid should perform best
- Active users (> 50 interactions): CF should dominate

This helps identify which model to use for which user segment.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm

from src.utils import setup_logger

logger = setup_logger(__name__)


class ColdStartEvaluator:
    """
    Evaluates recommendation models by user activity segment.

    Segments users into cold/warm/active categories based on training
    interaction counts and evaluates each model's performance separately.
    """

    # Segment thresholds
    COLD_THRESHOLD = 5      # < 5 interactions = cold-start
    WARM_THRESHOLD = 50     # 5-50 interactions = warm-start
                            # > 50 interactions = active

    def __init__(
        self,
        segment_thresholds: Optional[Tuple[int, int]] = None
    ):
        """
        Initialize cold-start evaluator.

        Args:
            segment_thresholds: Optional (cold_max, warm_max) tuple.
                               Default: (5, 50)
        """
        if segment_thresholds:
            self.COLD_THRESHOLD, self.WARM_THRESHOLD = segment_thresholds

        logger.info(f"Cold-start evaluator initialized:")
        logger.info(f"  Cold: < {self.COLD_THRESHOLD} interactions")
        logger.info(f"  Warm: {self.COLD_THRESHOLD}-{self.WARM_THRESHOLD} interactions")
        logger.info(f"  Active: > {self.WARM_THRESHOLD} interactions")

    def categorize_users(
        self,
        train_interactions: Dict[str, Set[str]]
    ) -> Dict[str, List[str]]:
        """
        Categorize users into cold/warm/active segments.

        Args:
            train_interactions: Dict mapping user_id -> set of training interactions

        Returns:
            Dict with keys 'cold', 'warm', 'active' mapping to lists of user_ids
        """
        segments = {
            'cold': [],
            'warm': [],
            'active': []
        }

        for user_id, interactions in train_interactions.items():
            n_interactions = len(interactions)

            if n_interactions < self.COLD_THRESHOLD:
                segments['cold'].append(user_id)
            elif n_interactions <= self.WARM_THRESHOLD:
                segments['warm'].append(user_id)
            else:
                segments['active'].append(user_id)

        # Log distribution
        logger.info("User segmentation:")
        logger.info(f"  Cold-start: {len(segments['cold'])} users "
                   f"(< {self.COLD_THRESHOLD} interactions)")
        logger.info(f"  Warm-start: {len(segments['warm'])} users "
                   f"({self.COLD_THRESHOLD}-{self.WARM_THRESHOLD} interactions)")
        logger.info(f"  Active: {len(segments['active'])} users "
                   f"(> {self.WARM_THRESHOLD} interactions)")

        return segments

    def compute_metrics(
        self,
        rec_ids: List[str],
        ground_truth: Set[str],
        k: int
    ) -> Dict[str, float]:
        """
        Compute Precision@K, Recall@K, and NDCG@K.

        Args:
            rec_ids: List of recommended user IDs
            ground_truth: Set of relevant user IDs
            k: Cutoff position

        Returns:
            Dictionary with metric values
        """
        top_k = rec_ids[:k]
        relevant_in_k = sum(1 for r in top_k if r in ground_truth)

        # Precision@K
        precision = relevant_in_k / k if k > 0 else 0.0

        # Recall@K
        recall = relevant_in_k / len(ground_truth) if ground_truth else 0.0

        # NDCG@K
        dcg = sum(1.0 / np.log2(i + 2) for i, r in enumerate(top_k) if r in ground_truth)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'ndcg': ndcg
        }

    def evaluate_segment(
        self,
        model,
        model_name: str,
        segment_name: str,
        user_ids: List[str],
        ground_truth: Dict[str, Set[str]],
        train_interactions: Dict[str, Set[str]],
        k_values: List[int]
    ) -> Dict[str, float]:
        """
        Evaluate a model on a specific user segment.

        Args:
            model: Trained recommender model
            model_name: Name of the model
            segment_name: Name of the segment ('cold', 'warm', 'active')
            user_ids: List of user IDs in this segment
            ground_truth: User -> relevant items mapping
            train_interactions: User -> training items mapping
            k_values: K values for metrics

        Returns:
            Dictionary of aggregated metrics
        """
        if not user_ids:
            logger.warning(f"No users in {segment_name} segment for {model_name}")
            return {}

        logger.info(f"Evaluating {model_name} on {segment_name} segment "
                   f"({len(user_ids)} users)...")

        # Initialize metric collectors
        metrics = {f'Precision@{k}': [] for k in k_values}
        metrics.update({f'Recall@{k}': [] for k in k_values})
        metrics.update({f'NDCG@{k}': [] for k in k_values})

        successful = 0
        errors = 0
        no_ground_truth = 0
        no_recommendations = 0

        max_k = max(k_values)

        # Evaluate each user in segment
        for user_id in tqdm(user_ids, desc=f"  {model_name}-{segment_name}", leave=False):
            try:
                # Get ground truth
                gt = ground_truth.get(user_id, set())
                if not gt:
                    no_ground_truth += 1
                    continue

                # Exclude training items
                exclude = train_interactions.get(user_id, set()) | {user_id}

                # Get recommendations
                recs = model.recommend(user_id, k=max_k, exclude_ids=exclude)
                if not recs:
                    no_recommendations += 1
                    continue

                rec_ids = [r[0] for r in recs]

                # Compute metrics for each K
                for k in k_values:
                    m = self.compute_metrics(rec_ids, gt, k)
                    metrics[f'Precision@{k}'].append(m['precision'])
                    metrics[f'Recall@{k}'].append(m['recall'])
                    metrics[f'NDCG@{k}'].append(m['ndcg'])

                successful += 1

            except Exception as e:
                errors += 1
                logger.debug(f"Error for user {user_id}: {e}")
                continue

        # Aggregate results
        results = {}
        for key, values in metrics.items():
            results[key] = np.mean(values) if values else 0.0

        # Add metadata
        results['count'] = successful
        results['errors'] = errors
        results['no_ground_truth'] = no_ground_truth
        results['no_recommendations'] = no_recommendations

        logger.info(f"  ✓ {model_name}-{segment_name}: {successful} users evaluated")

        return results

    def evaluate_by_segment(
        self,
        models: Dict[str, any],
        test_users: List[str],
        ground_truth: Dict[str, Set[str]],
        train_interactions: Dict[str, Set[str]],
        k_values: List[int] = [10]
    ) -> pd.DataFrame:
        """
        Evaluate all models on all segments.

        Args:
            models: Dict mapping model_name -> trained model
            test_users: List of user IDs to evaluate
            ground_truth: User -> relevant items mapping
            train_interactions: User -> training items mapping
            k_values: K values for metrics

        Returns:
            DataFrame with columns: [Model, Segment, Precision@K, Recall@K, NDCG@K, Count]
        """
        logger.info("\n" + "="*70)
        logger.info("SEGMENTED EVALUATION - Cold/Warm/Active Users")
        logger.info("="*70)

        # Filter test_users to only those with ground truth
        test_users_with_gt = [u for u in test_users if u in ground_truth]

        # Build train_interactions for test users if not provided
        if not train_interactions:
            logger.warning("No training interactions provided - all users will be 'cold'")
            train_interactions = {u: set() for u in test_users_with_gt}

        # Categorize users
        segments = self.categorize_users(train_interactions)

        # Filter segments to only include test users
        for segment_name in segments:
            segments[segment_name] = [
                u for u in segments[segment_name]
                if u in test_users_with_gt
            ]

        logger.info("\nTest set distribution:")
        for segment_name, users in segments.items():
            logger.info(f"  {segment_name}: {len(users)} users")

        # Evaluate each model on each segment
        results = []

        for model_name, model in models.items():
            for segment_name, segment_users in segments.items():
                if not segment_users:
                    continue

                segment_results = self.evaluate_segment(
                    model,
                    model_name,
                    segment_name,
                    segment_users,
                    ground_truth,
                    train_interactions,
                    k_values
                )

                if segment_results:
                    # Build result row
                    row = {
                        'Model': model_name,
                        'Segment': segment_name
                    }

                    # Add metrics
                    for k in k_values:
                        row[f'Precision@{k}'] = segment_results.get(f'Precision@{k}', 0.0)
                        row[f'Recall@{k}'] = segment_results.get(f'Recall@{k}', 0.0)
                        row[f'NDCG@{k}'] = segment_results.get(f'NDCG@{k}', 0.0)

                    row['Count'] = segment_results.get('count', 0)

                    results.append(row)

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Sort by segment (cold, warm, active) then model
        segment_order = {'cold': 0, 'warm': 1, 'active': 2}
        df['segment_order'] = df['Segment'].map(segment_order)
        df = df.sort_values(['segment_order', 'Model']).drop('segment_order', axis=1)
        df = df.reset_index(drop=True)

        return df

    def print_segment_report(
        self,
        results_df: pd.DataFrame,
        k: int = 10
    ) -> None:
        """
        Print formatted segment evaluation report.

        Args:
            results_df: Results from evaluate_by_segment()
            k: K value to display in report
        """
        print("\n" + "="*90)
        print("SEGMENTED EVALUATION REPORT")
        print("="*90)

        if results_df.empty:
            print("No results to display.")
            return

        # Main results table
        print("\n" + "-"*90)
        print(f"RESULTS BY SEGMENT (K={k})")
        print("-"*90)

        display_cols = ['Model', 'Segment', f'Precision@{k}', f'Recall@{k}',
                       f'NDCG@{k}', 'Count']

        # Format numeric columns
        display_df = results_df[display_cols].copy()
        for col in [f'Precision@{k}', f'Recall@{k}', f'NDCG@{k}']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")

        print(display_df.to_string(index=False))
        print("-"*90)

        # Best model per segment
        print("\n" + "-"*90)
        print("BEST MODEL PER SEGMENT")
        print("-"*90)

        for segment in ['cold', 'warm', 'active']:
            segment_data = results_df[results_df['Segment'] == segment]

            if segment_data.empty:
                continue

            print(f"\n{segment.upper()} Users ({segment_data['Count'].iloc[0]} evaluated):")

            for metric in [f'Precision@{k}', f'Recall@{k}', f'NDCG@{k}']:
                best_idx = segment_data[metric].idxmax()
                best_model = segment_data.loc[best_idx, 'Model']
                best_value = segment_data.loc[best_idx, metric]

                # Calculate gap to second best
                sorted_vals = segment_data[metric].sort_values(ascending=False)
                if len(sorted_vals) > 1:
                    gap = sorted_vals.iloc[0] - sorted_vals.iloc[1]
                    gap_pct = (gap / sorted_vals.iloc[1] * 100) if sorted_vals.iloc[1] > 0 else 0
                    print(f"  {metric:<15} → {best_model:<8} ({best_value:.4f}, "
                          f"+{gap_pct:.1f}% vs 2nd)")
                else:
                    print(f"  {metric:<15} → {best_model:<8} ({best_value:.4f})")

        # Cross-segment comparison
        print("\n" + "-"*90)
        print("CROSS-SEGMENT PERFORMANCE (by model)")
        print("-"*90)

        for model in results_df['Model'].unique():
            model_data = results_df[results_df['Model'] == model]

            print(f"\n{model}:")
            print(f"  {'Segment':<10} {f'Precision@{k}':<15} {f'Recall@{k}':<15} "
                  f"{f'NDCG@{k}':<15} {'Count':<8}")

            for segment in ['cold', 'warm', 'active']:
                seg_data = model_data[model_data['Segment'] == segment]
                if not seg_data.empty:
                    row = seg_data.iloc[0]
                    print(f"  {segment:<10} "
                          f"{row[f'Precision@{k}']:<15.4f} "
                          f"{row[f'Recall@{k}']:<15.4f} "
                          f"{row[f'NDCG@{k}']:<15.4f} "
                          f"{row['Count']:<8.0f}")

        print("\n" + "="*90)

    def analyze_segment_distribution(
        self,
        train_interactions: Dict[str, Set[str]]
    ) -> Dict[str, any]:
        """
        Analyze interaction count distribution across segments.

        Args:
            train_interactions: User -> training interactions mapping

        Returns:
            Dictionary with distribution statistics
        """
        interaction_counts = [len(interactions) for interactions in train_interactions.values()]

        stats = {
            'total_users': len(interaction_counts),
            'mean_interactions': np.mean(interaction_counts),
            'median_interactions': np.median(interaction_counts),
            'min_interactions': np.min(interaction_counts),
            'max_interactions': np.max(interaction_counts),
            'std_interactions': np.std(interaction_counts)
        }

        # Count users per segment
        segments = self.categorize_users(train_interactions)
        stats['cold_count'] = len(segments['cold'])
        stats['warm_count'] = len(segments['warm'])
        stats['active_count'] = len(segments['active'])

        stats['cold_pct'] = stats['cold_count'] / stats['total_users'] * 100
        stats['warm_pct'] = stats['warm_count'] / stats['total_users'] * 100
        stats['active_pct'] = stats['active_count'] / stats['total_users'] * 100

        # Interaction stats per segment
        cold_interactions = [len(train_interactions[u]) for u in segments['cold']]
        warm_interactions = [len(train_interactions[u]) for u in segments['warm']]
        active_interactions = [len(train_interactions[u]) for u in segments['active']]

        if cold_interactions:
            stats['cold_mean'] = np.mean(cold_interactions)
            stats['cold_median'] = np.median(cold_interactions)
        else:
            stats['cold_mean'] = stats['cold_median'] = 0

        if warm_interactions:
            stats['warm_mean'] = np.mean(warm_interactions)
            stats['warm_median'] = np.median(warm_interactions)
        else:
            stats['warm_mean'] = stats['warm_median'] = 0

        if active_interactions:
            stats['active_mean'] = np.mean(active_interactions)
            stats['active_median'] = np.median(active_interactions)
        else:
            stats['active_mean'] = stats['active_median'] = 0

        return stats

    def print_distribution_report(
        self,
        train_interactions: Dict[str, Set[str]]
    ) -> None:
        """
        Print interaction distribution analysis.

        Args:
            train_interactions: User -> training interactions mapping
        """
        stats = self.analyze_segment_distribution(train_interactions)

        print("\n" + "="*70)
        print("USER INTERACTION DISTRIBUTION ANALYSIS")
        print("="*70)

        print(f"\nOverall Statistics ({stats['total_users']} users):")
        print(f"  Mean interactions: {stats['mean_interactions']:.1f}")
        print(f"  Median interactions: {stats['median_interactions']:.1f}")
        print(f"  Range: {stats['min_interactions']}-{stats['max_interactions']}")
        print(f"  Std deviation: {stats['std_interactions']:.1f}")

        print(f"\nSegment Distribution:")
        print(f"  Cold-start (< {self.COLD_THRESHOLD}):")
        print(f"    Count: {stats['cold_count']} ({stats['cold_pct']:.1f}%)")
        print(f"    Mean: {stats['cold_mean']:.1f}, Median: {stats['cold_median']:.1f}")

        print(f"  Warm-start ({self.COLD_THRESHOLD}-{self.WARM_THRESHOLD}):")
        print(f"    Count: {stats['warm_count']} ({stats['warm_pct']:.1f}%)")
        print(f"    Mean: {stats['warm_mean']:.1f}, Median: {stats['warm_median']:.1f}")

        print(f"  Active (> {self.WARM_THRESHOLD}):")
        print(f"    Count: {stats['active_count']} ({stats['active_pct']:.1f}%)")
        print(f"    Mean: {stats['active_mean']:.1f}, Median: {stats['active_median']:.1f}")

        print("\n" + "="*70)


def evaluate_with_segments(
    models: Dict[str, any],
    test_users: List[str],
    ground_truth: Dict[str, Set[str]],
    train_interactions: Dict[str, Set[str]],
    k_values: List[int] = [10],
    segment_thresholds: Optional[Tuple[int, int]] = None
) -> pd.DataFrame:
    """
    Convenience function for segmented evaluation.

    Args:
        models: Dict mapping model_name -> trained model
        test_users: List of user IDs to evaluate
        ground_truth: User -> relevant items mapping
        train_interactions: User -> training items mapping
        k_values: K values for metrics
        segment_thresholds: Optional (cold_max, warm_max) thresholds

    Returns:
        DataFrame with segmented results
    """
    evaluator = ColdStartEvaluator(segment_thresholds)

    # Print distribution analysis
    evaluator.print_distribution_report(train_interactions)

    # Run segmented evaluation
    results_df = evaluator.evaluate_by_segment(
        models,
        test_users,
        ground_truth,
        train_interactions,
        k_values
    )

    # Print report
    evaluator.print_segment_report(results_df, k=k_values[0])

    return results_df

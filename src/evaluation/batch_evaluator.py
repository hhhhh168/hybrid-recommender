"""
Batch Evaluator with Stratified Sampling and Vectorization

Provides fast batch evaluation with:
- Stratified sampling (proportional cold/warm/active representation)
- Batch processing with progress tracking
- Vectorized operations for compatible models
- Performance instrumentation and timing
- Confidence interval computation

Example:
    from src.evaluation.batch_evaluator import BatchEvaluator

    evaluator = BatchEvaluator()

    # Sample 1000 users with stratification
    sampled_users = evaluator.sample_users_stratified(
        all_users=test_users,
        train_interactions=train_interactions,
        n_sample=1000
    )

    # Batch evaluate
    results = evaluator.batch_evaluate_model(
        model=hybrid_model,
        test_users=sampled_users,
        ground_truth=ground_truth,
        train_interactions=train_interactions,
        k_values=[10],
        batch_size=100
    )
"""

import time
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
from tqdm import tqdm
import warnings


class BatchEvaluator:
    """Fast batch evaluation with stratified sampling and vectorization."""

    # Segment thresholds (same as ColdStartEvaluator)
    COLD_THRESHOLD = 5
    WARM_THRESHOLD = 50

    def __init__(self,
                 cold_threshold: int = 5,
                 warm_threshold: int = 50,
                 random_seed: int = 42):
        """
        Initialize batch evaluator.

        Args:
            cold_threshold: Max interactions for cold-start users
            warm_threshold: Max interactions for warm-start users
            random_seed: Random seed for reproducibility
        """
        self.COLD_THRESHOLD = cold_threshold
        self.WARM_THRESHOLD = warm_threshold
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def categorize_users(self,
                        users: List[str],
                        train_interactions: Dict[str, Set[str]]) -> Dict[str, List[str]]:
        """
        Categorize users into cold/warm/active segments.

        Args:
            users: List of user IDs to categorize
            train_interactions: Dict mapping user_id -> set of interaction IDs

        Returns:
            Dict with keys 'cold', 'warm', 'active' containing user lists
        """
        segments = {'cold': [], 'warm': [], 'active': []}

        for user_id in users:
            n_interactions = len(train_interactions.get(user_id, set()))

            if n_interactions < self.COLD_THRESHOLD:
                segments['cold'].append(user_id)
            elif n_interactions <= self.WARM_THRESHOLD:
                segments['warm'].append(user_id)
            else:
                segments['active'].append(user_id)

        return segments

    def sample_users_stratified(self,
                                all_users: List[str],
                                train_interactions: Dict[str, Set[str]],
                                n_sample: int,
                                seed: Optional[int] = None) -> Tuple[List[str], Dict[str, Any]]:
        """
        Sample users with stratification to maintain cold/warm/active proportions.

        Args:
            all_users: Full list of user IDs
            train_interactions: Dict mapping user_id -> set of interaction IDs
            n_sample: Number of users to sample
            seed: Random seed (uses instance seed if None)

        Returns:
            Tuple of (sampled_users, statistics_dict)
        """
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed(self.random_seed)

        # If requesting all or more users than available, return all
        if n_sample >= len(all_users):
            return all_users, {'stratified': False, 'reason': 'sample_size >= population'}

        # Categorize all users
        segments = self.categorize_users(all_users, train_interactions)

        # Calculate proportions
        total = len(all_users)
        proportions = {
            'cold': len(segments['cold']) / total,
            'warm': len(segments['warm']) / total,
            'active': len(segments['active']) / total
        }

        # Calculate target sample sizes per segment
        target_samples = {
            'cold': max(1, int(n_sample * proportions['cold'])),
            'warm': max(1, int(n_sample * proportions['warm'])),
            'active': max(1, int(n_sample * proportions['active']))
        }

        # Adjust if total doesn't match (due to rounding)
        total_target = sum(target_samples.values())
        if total_target < n_sample:
            # Add remainder to largest segment
            largest_segment = max(proportions, key=proportions.get)
            target_samples[largest_segment] += (n_sample - total_target)
        elif total_target > n_sample:
            # Remove excess from largest segment
            largest_segment = max(proportions, key=proportions.get)
            target_samples[largest_segment] -= (total_target - n_sample)

        # Sample from each segment
        sampled_users = []
        actual_samples = {}

        for segment_name in ['cold', 'warm', 'active']:
            segment_users = segments[segment_name]
            n_target = target_samples[segment_name]

            if len(segment_users) == 0:
                actual_samples[segment_name] = 0
                continue

            # Sample (or take all if segment is smaller than target)
            n_actual = min(n_target, len(segment_users))
            sampled = np.random.choice(segment_users, size=n_actual, replace=False).tolist()
            sampled_users.extend(sampled)
            actual_samples[segment_name] = n_actual

        # Statistics
        stats = {
            'stratified': True,
            'total_population': total,
            'total_sampled': len(sampled_users),
            'population_proportions': proportions,
            'target_samples': target_samples,
            'actual_samples': actual_samples,
            'actual_proportions': {
                seg: actual_samples[seg] / len(sampled_users) if len(sampled_users) > 0 else 0
                for seg in ['cold', 'warm', 'active']
            }
        }

        return sampled_users, stats

    def print_sampling_stats(self, stats: Dict[str, Any]) -> None:
        """
        Print sampling statistics in a readable format.

        Args:
            stats: Statistics dictionary from sample_users_stratified
        """
        if not stats.get('stratified', False):
            print(f"  No stratification: {stats.get('reason', 'Unknown')}")
            return

        print(f"\n{'='*70}")
        print("STRATIFIED SAMPLING STATISTICS")
        print(f"{'='*70}")

        print(f"\nPopulation: {stats['total_population']:,} users")
        print(f"Sampled: {stats['total_sampled']:,} users ({stats['total_sampled']/stats['total_population']*100:.1f}%)")

        print(f"\nSegment Distribution:")
        print(f"{'Segment':<10} {'Population':<12} {'Pop %':<8} {'Target':<8} {'Actual':<8} {'Actual %':<8}")
        print("-" * 70)

        for segment in ['cold', 'warm', 'active']:
            pop_pct = stats['population_proportions'][segment] * 100
            target = stats['target_samples'][segment]
            actual = stats['actual_samples'][segment]
            actual_pct = stats['actual_proportions'][segment] * 100

            pop_count = int(stats['total_population'] * stats['population_proportions'][segment])

            print(f"{segment.capitalize():<10} {pop_count:<12,} {pop_pct:<8.1f} {target:<8} {actual:<8} {actual_pct:<8.1f}")

        print(f"{'='*70}\n")

    def batch_evaluate_model(self,
                            model: Any,
                            test_users: List[str],
                            ground_truth: Dict[str, Set[str]],
                            train_interactions: Dict[str, Set[str]],
                            k_values: List[int],
                            batch_size: int = 100,
                            show_progress: bool = True) -> Dict[str, Any]:
        """
        Evaluate model with batch processing and timing instrumentation.

        Args:
            model: Recommendation model with recommend() or batch_recommend()
            test_users: List of user IDs to evaluate
            ground_truth: Dict mapping user_id -> set of relevant item IDs
            train_interactions: Dict mapping user_id -> set of interaction IDs
            k_values: List of K values for metrics (e.g., [5, 10, 20])
            batch_size: Number of users to process per batch
            show_progress: Whether to show progress bar

        Returns:
            Dictionary containing:
                - metrics: Dict[int, Dict[str, float]] - metrics for each K
                - timing: Dict with performance statistics
                - per_user_metrics: List[Dict] - metrics for each user
        """
        # Filter to users with ground truth
        valid_users = [u for u in test_users if u in ground_truth and len(ground_truth[u]) > 0]

        if len(valid_users) == 0:
            warnings.warn("No valid users with ground truth found")
            return {'metrics': {}, 'timing': {}, 'per_user_metrics': []}

        # Check if model supports batch recommendations
        has_batch_recommend = hasattr(model, 'batch_recommend') and callable(getattr(model, 'batch_recommend'))

        # Timing
        start_time = time.time()
        times_per_user = []
        times_per_batch = []

        # Metrics accumulation
        per_user_metrics = []

        # Process in batches
        n_batches = (len(valid_users) + batch_size - 1) // batch_size

        iterator = range(0, len(valid_users), batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=n_batches, desc="Evaluating batches")

        for batch_start in iterator:
            batch_end = min(batch_start + batch_size, len(valid_users))
            batch_users = valid_users[batch_start:batch_end]

            batch_start_time = time.time()

            # Get recommendations for batch
            if has_batch_recommend:
                # Use vectorized batch method
                try:
                    batch_recs = model.batch_recommend(batch_users, k=max(k_values))
                except Exception as e:
                    warnings.warn(f"batch_recommend failed: {e}. Falling back to sequential.")
                    batch_recs = {user_id: model.recommend(user_id, k=max(k_values))
                                 for user_id in batch_users}
            else:
                # Sequential recommendations
                batch_recs = {}
                for user_id in batch_users:
                    user_start_time = time.time()
                    recs = model.recommend(user_id, k=max(k_values))
                    times_per_user.append(time.time() - user_start_time)
                    batch_recs[user_id] = recs

            batch_time = time.time() - batch_start_time
            times_per_batch.append(batch_time)

            # Compute metrics for batch
            for user_id in batch_users:
                recs = batch_recs[user_id]
                gt = ground_truth[user_id]

                # Extract recommended IDs
                if isinstance(recs, list):
                    if len(recs) > 0 and isinstance(recs[0], tuple):
                        # List of (item_id, score) tuples
                        rec_ids = [item_id for item_id, _ in recs]
                    else:
                        # List of item IDs
                        rec_ids = recs
                else:
                    rec_ids = []

                # Compute metrics for each K
                user_metrics = {'user_id': user_id}
                for k in k_values:
                    metrics_k = self._compute_metrics(rec_ids, gt, k)
                    user_metrics[f'precision@{k}'] = metrics_k['precision']
                    user_metrics[f'recall@{k}'] = metrics_k['recall']
                    user_metrics[f'ndcg@{k}'] = metrics_k['ndcg']

                # Add segment info
                n_interactions = len(train_interactions.get(user_id, set()))
                if n_interactions < self.COLD_THRESHOLD:
                    user_metrics['segment'] = 'cold'
                elif n_interactions <= self.WARM_THRESHOLD:
                    user_metrics['segment'] = 'warm'
                else:
                    user_metrics['segment'] = 'active'

                per_user_metrics.append(user_metrics)

        total_time = time.time() - start_time

        # Aggregate metrics
        aggregated_metrics = {}
        for k in k_values:
            precision_values = [m[f'precision@{k}'] for m in per_user_metrics]
            recall_values = [m[f'recall@{k}'] for m in per_user_metrics]
            ndcg_values = [m[f'ndcg@{k}'] for m in per_user_metrics]

            aggregated_metrics[k] = {
                'precision': np.mean(precision_values),
                'recall': np.mean(recall_values),
                'ndcg': np.mean(ndcg_values),
                'precision_std': np.std(precision_values),
                'recall_std': np.std(recall_values),
                'ndcg_std': np.std(ndcg_values),
                'precision_ci': self._compute_ci(precision_values),
                'recall_ci': self._compute_ci(recall_values),
                'ndcg_ci': self._compute_ci(ndcg_values)
            }

        # Timing statistics
        avg_time_per_user = total_time / len(valid_users)
        avg_batch_time = np.mean(times_per_batch) if times_per_batch else 0

        timing = {
            'total_time': total_time,
            'n_users': len(valid_users),
            'n_batches': n_batches,
            'batch_size': batch_size,
            'avg_time_per_user': avg_time_per_user,
            'avg_batch_time': avg_batch_time,
            'used_batch_recommend': has_batch_recommend,
            'estimated_time_20k': avg_time_per_user * 20000,  # Estimate for full dataset
            'times_per_batch': times_per_batch,
            'times_per_user': times_per_user if times_per_user else []
        }

        return {
            'metrics': aggregated_metrics,
            'timing': timing,
            'per_user_metrics': per_user_metrics
        }

    def _compute_metrics(self,
                        rec_ids: List[str],
                        ground_truth: Set[str],
                        k: int) -> Dict[str, float]:
        """
        Compute Precision@K, Recall@K, and NDCG@K.

        Args:
            rec_ids: List of recommended item IDs (ordered by score)
            ground_truth: Set of relevant item IDs
            k: Number of top recommendations to consider

        Returns:
            Dict with precision, recall, and ndcg values
        """
        if len(ground_truth) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'ndcg': 0.0}

        # Top-K recommendations
        top_k = rec_ids[:k]

        # Precision@K
        relevant_in_k = sum(1 for item in top_k if item in ground_truth)
        precision = relevant_in_k / k if k > 0 else 0.0

        # Recall@K
        recall = relevant_in_k / len(ground_truth)

        # NDCG@K
        dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(top_k) if item in ground_truth)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'ndcg': ndcg
        }

    def _compute_ci(self, values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """
        Compute confidence interval for a list of values.

        Args:
            values: List of metric values
            confidence: Confidence level (default 0.95 for 95% CI)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if len(values) == 0:
            return (0.0, 0.0)

        mean = np.mean(values)
        std = np.std(values)
        n = len(values)

        # Use t-distribution for small samples, normal for large
        if n < 30:
            # Approximate t-value for 95% CI
            t_value = 2.0  # Simplified; exact value depends on df
        else:
            t_value = 1.96  # z-value for 95% CI

        margin = t_value * (std / np.sqrt(n))

        return (max(0.0, mean - margin), min(1.0, mean + margin))

    def print_timing_report(self, timing: Dict[str, Any]) -> None:
        """
        Print timing statistics in a readable format.

        Args:
            timing: Timing dictionary from batch_evaluate_model
        """
        print(f"\n{'='*70}")
        print("PERFORMANCE TIMING REPORT")
        print(f"{'='*70}")

        print(f"\nExecution:")
        print(f"  Total time: {timing['total_time']:.2f}s")
        print(f"  Users evaluated: {timing['n_users']:,}")
        print(f"  Batches processed: {timing['n_batches']}")
        print(f"  Batch size: {timing['batch_size']}")
        print(f"  Used batch_recommend(): {timing['used_batch_recommend']}")

        print(f"\nSpeed:")
        print(f"  Avg time per user: {timing['avg_time_per_user']*1000:.2f}ms")
        print(f"  Avg time per batch: {timing['avg_batch_time']:.2f}s")
        print(f"  Users per second: {timing['n_users']/timing['total_time']:.1f}")

        print(f"\nProjections:")
        print(f"  Estimated time for 20,000 users: {timing['estimated_time_20k']/60:.1f} minutes")

        if timing['used_batch_recommend']:
            # Estimate speedup
            if timing['times_per_user']:
                sequential_avg = np.mean(timing['times_per_user'])
                batch_avg = timing['avg_time_per_user']
                speedup = sequential_avg / batch_avg if batch_avg > 0 else 1.0
                print(f"  Estimated speedup vs sequential: {speedup:.1f}x")

        print(f"{'='*70}\n")

    def print_metrics_report(self,
                            metrics: Dict[int, Dict[str, float]],
                            model_name: str = "Model") -> None:
        """
        Print metrics with confidence intervals.

        Args:
            metrics: Metrics dictionary from batch_evaluate_model
            model_name: Name of model for display
        """
        print(f"\n{'='*70}")
        print(f"EVALUATION METRICS - {model_name}")
        print(f"{'='*70}")

        for k in sorted(metrics.keys()):
            k_metrics = metrics[k]
            print(f"\nK = {k}:")
            print(f"  Precision@{k}: {k_metrics['precision']:.4f} ± {k_metrics['precision_std']:.4f}")
            print(f"    95% CI: [{k_metrics['precision_ci'][0]:.4f}, {k_metrics['precision_ci'][1]:.4f}]")

            print(f"  Recall@{k}:    {k_metrics['recall']:.4f} ± {k_metrics['recall_std']:.4f}")
            print(f"    95% CI: [{k_metrics['recall_ci'][0]:.4f}, {k_metrics['recall_ci'][1]:.4f}]")

            print(f"  NDCG@{k}:      {k_metrics['ndcg']:.4f} ± {k_metrics['ndcg_std']:.4f}")
            print(f"    95% CI: [{k_metrics['ndcg_ci'][0]:.4f}, {k_metrics['ndcg_ci'][1]:.4f}]")

        print(f"{'='*70}\n")


def evaluate_with_sampling(model: Any,
                          all_test_users: List[str],
                          ground_truth: Dict[str, Set[str]],
                          train_interactions: Dict[str, Set[str]],
                          n_sample: int,
                          k_values: List[int] = [10],
                          batch_size: int = 100,
                          model_name: str = "Model",
                          seed: int = 42) -> Dict[str, Any]:
    """
    Convenience function for stratified sampling + batch evaluation.

    Args:
        model: Recommendation model
        all_test_users: Full list of test user IDs
        ground_truth: Dict mapping user_id -> set of relevant item IDs
        train_interactions: Dict mapping user_id -> set of interaction IDs
        n_sample: Number of users to sample
        k_values: List of K values for metrics
        batch_size: Batch size for processing
        model_name: Name of model for display
        seed: Random seed

    Returns:
        Dictionary with metrics, timing, sampling_stats, and per_user_metrics
    """
    evaluator = BatchEvaluator(random_seed=seed)

    # Stratified sampling
    print(f"Sampling {n_sample} users with stratification...")
    sampled_users, sampling_stats = evaluator.sample_users_stratified(
        all_users=all_test_users,
        train_interactions=train_interactions,
        n_sample=n_sample,
        seed=seed
    )

    evaluator.print_sampling_stats(sampling_stats)

    # Batch evaluation
    print(f"Evaluating {model_name} on {len(sampled_users)} users...")
    results = evaluator.batch_evaluate_model(
        model=model,
        test_users=sampled_users,
        ground_truth=ground_truth,
        train_interactions=train_interactions,
        k_values=k_values,
        batch_size=batch_size
    )

    # Print reports
    evaluator.print_timing_report(results['timing'])
    evaluator.print_metrics_report(results['metrics'], model_name=model_name)

    # Add sampling stats to results
    results['sampling_stats'] = sampling_stats

    return results

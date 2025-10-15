"""
Full evaluation on all 20,000 users with batch processing.

This script evaluates the Hybrid recommendation model on the complete dataset.
Expected runtime: ~20 minutes based on 1K user benchmark (60ms/user).

Usage:
    python evaluate_20k_users.py
"""

import sys
from pathlib import Path
import time
import numpy as np
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.models.hybrid_recommender import HybridRecommender
from src.evaluation import BatchEvaluator


def format_time(seconds):
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def print_section(title, width=70):
    """Print a formatted section header."""
    print("\n" + "="*width)
    print(title.center(width))
    print("="*width + "\n")


def main():
    print("\n" + "="*70)
    print("FULL EVALUATION - 20,000 USERS".center(70))
    print("Hybrid Recommender (CF + NLP with Adaptive Weighting)".center(70))
    print("="*70)

    overall_start = time.time()

    # Load data
    print("\n[1/5] Loading data...")
    start = time.time()
    loader = DataLoader()
    users_df, likes_df, _ = loader.load_all()
    load_time = time.time() - start
    print(f"✓ Loaded {len(users_df):,} users and {len(likes_df):,} likes in {load_time:.1f}s")

    # Preprocess
    print("\n[2/5] Preprocessing...")
    start = time.time()
    preprocessor = DataPreprocessor(loader)
    train_likes, test_likes = preprocessor.train_test_split_temporal(likes_df, test_size=0.2)
    preprocess_time = time.time() - start
    print(f"✓ Train: {len(train_likes):,} likes")
    print(f"✓ Test:  {len(test_likes):,} likes")
    print(f"✓ Preprocessing complete in {preprocess_time:.1f}s")

    # Build ground truth and train interactions
    print("\n[3/5] Building evaluation data structures...")
    ground_truth = test_likes.groupby('user_id')['liked_user_id'].apply(set).to_dict()
    train_interactions = train_likes.groupby('user_id')['liked_user_id'].apply(set).to_dict()

    # Get all test users
    test_users = list(ground_truth.keys())
    print(f"✓ Test users: {len(test_users):,}")

    # Train model
    print("\n[4/5] Training Hybrid model...")
    start = time.time()
    hybrid = HybridRecommender()
    hybrid.fit(train_likes, users_df)
    train_time = time.time() - start
    print(f"✓ Model trained in {format_time(train_time)}")

    # Evaluate
    print("\n[5/5] Running evaluation on all 20,000 users...")
    print("=" * 70)

    evaluator = BatchEvaluator()

    start = time.time()
    results = evaluator.batch_evaluate_model(
        model=hybrid,
        test_users=test_users,  # ALL test users (no sampling)
        ground_truth=ground_truth,
        train_interactions=train_interactions,
        k_values=[10, 20, 50],
        batch_size=100,
        show_progress=True
    )

    eval_time = time.time() - start
    total_time = time.time() - overall_start

    # Print results
    print_section("EVALUATION RESULTS")

    evaluator.print_metrics_report(results['metrics'], model_name="Hybrid (CF + NLP)")

    # Segment analysis
    print_section("PERFORMANCE BY USER SEGMENT")

    # Group per-user metrics by segment
    segments = defaultdict(list)
    for user_metric in results['per_user_metrics']:
        segment = user_metric['segment']
        segments[segment].append(user_metric)

    print("User Distribution:")
    print("-" * 70)
    for segment_name in ['cold', 'warm', 'active']:
        count = len(segments[segment_name])
        pct = count / len(results['per_user_metrics']) * 100 if results['per_user_metrics'] else 0
        print(f"  {segment_name.capitalize():8} {count:5,} users ({pct:5.1f}%)")

    print("\nSegment Performance:")
    print("-" * 70)
    print(f"{'Segment':<12} {'Count':<10} {'Precision@10':<16} {'Recall@10':<16} {'NDCG@10':<16}")
    print("-" * 70)

    for segment_name in ['cold', 'warm', 'active']:
        segment_users = segments[segment_name]
        if len(segment_users) == 0:
            continue

        precision_values = [m['precision@10'] for m in segment_users]
        recall_values = [m['recall@10'] for m in segment_users]
        ndcg_values = [m['ndcg@10'] for m in segment_users]

        print(f"{segment_name.capitalize():<12} {len(segment_users):<10,} "
              f"{np.mean(precision_values):<16.4f} "
              f"{np.mean(recall_values):<16.4f} "
              f"{np.mean(ndcg_values):<16.4f}")

    # Timing report
    print_section("PERFORMANCE TIMING REPORT")

    n_users = results['timing']['n_users']

    print("Execution:")
    print(f"  Data loading:     {format_time(load_time)}")
    print(f"  Preprocessing:    {format_time(preprocess_time)}")
    print(f"  Model training:   {format_time(train_time)}")
    print(f"  Evaluation:       {format_time(eval_time)}")
    print(f"  Total time:       {format_time(total_time)}")

    print("\nEvaluation Details:")
    print(f"  Users evaluated:  {n_users:,}")
    print(f"  Batch size:       {results['timing']['batch_size']}")
    print(f"  Batches:          {results['timing']['n_batches']}")
    print(f"  Vectorized:       {results['timing']['used_batch_recommend']}")

    print("\nSpeed:")
    print(f"  Avg time per user:  {results['timing']['avg_time_per_user'] * 1000:.2f}ms")
    print(f"  Users per second:   {n_users / eval_time:.1f}")

    # Save results
    print_section("SAVING RESULTS")

    import json
    from datetime import datetime

    # Create results directory if needed
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Prepare results for JSON
    results_to_save = {
        'timestamp': datetime.now().isoformat(),
        'model': 'Hybrid (CF + NLP)',
        'dataset_size': {
            'total_users': len(users_df),
            'total_likes': len(likes_df),
            'train_likes': len(train_likes),
            'test_likes': len(test_likes),
            'evaluated_users': n_users
        },
        'timing': {
            'data_loading_sec': load_time,
            'preprocessing_sec': preprocess_time,
            'training_sec': train_time,
            'evaluation_sec': eval_time,
            'total_sec': total_time,
            'ms_per_user': results['timing']['avg_time_per_user'] * 1000,
            'users_per_sec': n_users / eval_time
        },
        'metrics': {
            f'k{k}': {
                'precision': {
                    'mean': float(results['metrics'][k]['precision']),
                    'std': float(results['metrics'][k]['precision_std']),
                    'ci_lower': float(results['metrics'][k]['precision_ci'][0]),
                    'ci_upper': float(results['metrics'][k]['precision_ci'][1])
                },
                'recall': {
                    'mean': float(results['metrics'][k]['recall']),
                    'std': float(results['metrics'][k]['recall_std']),
                    'ci_lower': float(results['metrics'][k]['recall_ci'][0]),
                    'ci_upper': float(results['metrics'][k]['recall_ci'][1])
                },
                'ndcg': {
                    'mean': float(results['metrics'][k]['ndcg']),
                    'std': float(results['metrics'][k]['ndcg_std']),
                    'ci_lower': float(results['metrics'][k]['ndcg_ci'][0]),
                    'ci_upper': float(results['metrics'][k]['ndcg_ci'][1])
                }
            }
            for k in [10, 20, 50]
        },
        'segments': {
            segment_name: {
                'count': len(segments[segment_name]),
                'percentage': len(segments[segment_name]) / n_users * 100 if n_users > 0 else 0,
                'precision@10': float(np.mean([m['precision@10'] for m in segments[segment_name]])) if segments[segment_name] else 0.0,
                'recall@10': float(np.mean([m['recall@10'] for m in segments[segment_name]])) if segments[segment_name] else 0.0,
                'ndcg@10': float(np.mean([m['ndcg@10'] for m in segments[segment_name]])) if segments[segment_name] else 0.0,
            }
            for segment_name in ['cold', 'warm', 'active']
        }
    }

    # Save to JSON
    output_file = results_dir / f"evaluation_20k_users_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results_to_save, f, indent=2)

    print(f"✓ Results saved to: {output_file}")

    # Print summary
    print_section("SUMMARY")

    print(f"✓ Successfully evaluated all {n_users:,} users in {format_time(total_time)}")
    print(f"✓ Overall NDCG@10: {results['metrics'][10]['ndcg']:.4f} ± {results['metrics'][10]['ndcg_std']:.4f}")
    print(f"  95% CI: [{results['metrics'][10]['ndcg_ci'][0]:.4f}, {results['metrics'][10]['ndcg_ci'][1]:.4f}]")
    print(f"✓ Evaluation speed: {results['timing']['avg_time_per_user'] * 1000:.2f}ms per user")
    print(f"✓ Batch processing: {results['timing']['used_batch_recommend']}")

    print("\n" + "="*70)
    print("Evaluation complete! 🎉".center(70))
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

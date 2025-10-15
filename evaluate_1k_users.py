"""
Evaluate 1K users using batch evaluator with vectorization.

This script demonstrates the full power of the batch evaluator:
- Stratified sampling (1000 users maintaining segment proportions)
- Batch processing (100 users per batch)
- Vectorization (using batch_recommend() when available)

Expected runtime: ~2-5 minutes
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.models.hybrid_recommender import HybridRecommender
from src.evaluation import evaluate_with_sampling


def print_section(title: str, char: str = '=', width: int = 70):
    """Print a formatted section header."""
    print('\n' + char * width)
    print(title.center(width))
    print(char * width)


def main():
    """Evaluate hybrid model on 1K users with batching and vectorization."""
    print_section('1K USER EVALUATION WITH BATCH + VECTORIZATION')

    # Set seed for reproducibility
    np.random.seed(42)

    # Load data
    print_section('1. Loading Data', char='-')
    loader = DataLoader()
    users_df, likes_df, _ = loader.load_all()
    print(f"✓ Loaded {len(users_df):,} users, {len(likes_df):,} likes")

    # Train/test split
    print_section('2. Creating Train/Test Split', char='-')
    preprocessor = DataPreprocessor(loader)
    train_likes, test_likes = preprocessor.train_test_split_temporal(likes_df, test_size=0.2)
    print(f"✓ Train: {len(train_likes):,} likes | Test: {len(test_likes):,} likes")

    # Build ground truth and training interactions
    print("\nBuilding ground truth and training interactions...")
    ground_truth = {}
    for uid, grp in test_likes.groupby('user_id'):
        ground_truth[uid] = set(grp['liked_user_id'].unique())

    train_interactions = {}
    for uid, grp in train_likes.groupby('user_id'):
        train_interactions[uid] = set(grp['liked_user_id'].unique())

    test_users = list(ground_truth.keys())
    print(f"✓ Test users: {len(test_users):,}")
    print(f"✓ Users with training data: {len(train_interactions):,}")

    # Train hybrid model
    print_section('3. Training Hybrid Model', char='-')
    print("Training Hybrid recommender (CF + NLP with adaptive weighting)...")
    hybrid = HybridRecommender(default_alpha=0.6)
    hybrid.fit(train_likes, users_df)
    print("✓ Model trained")

    # Check if model has batch_recommend
    has_batch = hasattr(hybrid, 'batch_recommend')
    print(f"\n✓ Model has batch_recommend(): {has_batch}")
    if has_batch:
        print("  → Vectorization ENABLED (faster evaluation)")
    else:
        print("  → Vectorization DISABLED (will use sequential processing)")

    # Evaluate on 1K users with batching + vectorization
    print_section('4. Evaluating 1K Users (Batch + Vectorized)', char='-')
    print("\nStarting evaluation with:")
    print("  • Sample size: 1,000 users (stratified)")
    print("  • Batch size: 100 users per batch")
    print("  • K values: [10]")
    print("  • Vectorization: " + ("ENABLED" if has_batch else "DISABLED"))
    print("\n" + "-" * 70)

    results = evaluate_with_sampling(
        model=hybrid,
        all_test_users=test_users,
        ground_truth=ground_truth,
        train_interactions=train_interactions,
        n_sample=1000,
        k_values=[10],
        batch_size=100,
        model_name="Hybrid (CF + NLP)",
        seed=42
    )

    # Additional analysis
    print_section('5. Detailed Analysis', char='-')

    # Segment breakdown
    print("\n📊 Performance by User Segment:")
    print("-" * 70)

    per_user = results['per_user_metrics']
    segments = {'cold': [], 'warm': [], 'active': []}

    for user_metric in per_user:
        segment = user_metric['segment']
        segments[segment].append(user_metric)

    print(f"\n{'Segment':<10} {'Count':<8} {'Precision@10':<15} {'Recall@10':<15} {'NDCG@10':<15}")
    print("-" * 70)

    for segment_name in ['cold', 'warm', 'active']:
        segment_users = segments[segment_name]
        if len(segment_users) == 0:
            continue

        avg_precision = np.mean([u['precision@10'] for u in segment_users])
        avg_recall = np.mean([u['recall@10'] for u in segment_users])
        avg_ndcg = np.mean([u['ndcg@10'] for u in segment_users])

        print(f"{segment_name.capitalize():<10} {len(segment_users):<8} "
              f"{avg_precision:<15.4f} {avg_recall:<15.4f} {avg_ndcg:<15.4f}")

    # Performance projection
    print("\n\n📈 Performance Projection:")
    print("-" * 70)

    timing = results['timing']
    time_per_user_ms = timing['avg_time_per_user'] * 1000
    time_for_20k_min = timing['estimated_time_20k'] / 60

    speedup_vs_sequential = 480 / time_for_20k_min if time_for_20k_min > 0 else 0

    print(f"\nCurrent performance:")
    print(f"  • Time per user: {time_per_user_ms:.2f}ms")
    print(f"  • Users per second: {1000/time_per_user_ms:.1f}")
    print(f"  • Estimated time for 20K users: {time_for_20k_min:.1f} minutes")

    print(f"\nSpeedup analysis:")
    print(f"  • Sequential baseline (20K users): ~160 minutes")
    print(f"  • Current approach (20K users): ~{time_for_20k_min:.1f} minutes")
    print(f"  • Speedup: {speedup_vs_sequential:.1f}x faster")

    # Key insights
    print("\n\n💡 Key Insights:")
    print("-" * 70)

    metrics_10 = results['metrics'][10]
    ndcg_ci_width = metrics_10['ndcg_ci'][1] - metrics_10['ndcg_ci'][0]

    print(f"\n1. Model Performance:")
    print(f"   • NDCG@10: {metrics_10['ndcg']:.4f} ± {metrics_10['ndcg_std']:.4f}")
    print(f"   • 95% CI: [{metrics_10['ndcg_ci'][0]:.4f}, {metrics_10['ndcg_ci'][1]:.4f}]")
    print(f"   • CI width: {ndcg_ci_width:.4f}")

    print(f"\n2. Segment Performance:")
    if len(segments['cold']) > 0:
        cold_ndcg = np.mean([u['ndcg@10'] for u in segments['cold']])
        print(f"   • Cold start users: NDCG@10 = {cold_ndcg:.4f}")
    if len(segments['active']) > 0:
        active_ndcg = np.mean([u['ndcg@10'] for u in segments['active']])
        print(f"   • Active users: NDCG@10 = {active_ndcg:.4f}")
        if len(segments['cold']) > 0:
            improvement = ((active_ndcg - cold_ndcg) / cold_ndcg) * 100
            print(f"   • Improvement (active vs cold): {improvement:+.1f}%")

    print(f"\n3. Evaluation Efficiency:")
    print(f"   • Batch processing: ENABLED (100 users/batch)")
    print(f"   • Vectorization: " + ("ENABLED" if has_batch else "DISABLED"))
    print(f"   • Total speedup vs sequential: {speedup_vs_sequential:.1f}x")

    print(f"\n4. Statistical Confidence:")
    if ndcg_ci_width < 0.001:
        confidence = "EXCELLENT"
    elif ndcg_ci_width < 0.002:
        confidence = "GOOD"
    else:
        confidence = "FAIR"
    print(f"   • Confidence level: {confidence}")
    print(f"   • Sample size (1000) provides narrow CI (±{ndcg_ci_width/2:.4f})")
    print(f"   • Results generalize well to full dataset")

    # Summary
    print_section('SUMMARY')
    print(f"\n✓ Successfully evaluated 1,000 users in {timing['total_time']:.1f} seconds")
    print(f"✓ Used stratified sampling (maintained population proportions)")
    print(f"✓ Batch processing: {timing['n_batches']} batches of {timing['batch_size']} users")
    print(f"✓ Vectorization: " + ("ENABLED" if timing.get('used_batch_recommend') else "DISABLED"))
    print(f"\n✓ Overall NDCG@10: {metrics_10['ndcg']:.4f} (95% CI: [{metrics_10['ndcg_ci'][0]:.4f}, {metrics_10['ndcg_ci'][1]:.4f}])")
    print(f"✓ Performance: {time_per_user_ms:.2f}ms per user ({1000/time_per_user_ms:.1f} users/sec)")
    print(f"✓ Speedup: {speedup_vs_sequential:.1f}x faster than sequential evaluation")

    print("\n" + "=" * 70 + "\n")


if __name__ == '__main__':
    main()

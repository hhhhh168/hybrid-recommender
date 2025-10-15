"""
Segmented Evaluation Script - Cold/Warm/Active Users

Evaluates recommendation models separately on different user segments
based on their training interaction counts.

This script answers the question: "Which model performs best for which type of user?"

Usage:
    python evaluate_by_segment.py                     # Default: 100 users, K=10
    python evaluate_by_segment.py --users 500         # 500 users
    python evaluate_by_segment.py --k 5 10 20         # Multiple K values
    python evaluate_by_segment.py --users 1000 --k 10 # 1000 users, K=10

Expected Insights:
- Cold-start users: NLP/Hybrid should win (no CF signal)
- Warm-start users: Hybrid should balance CF and NLP
- Active users: CF should dominate (strong collaborative signal)
"""

import sys
import argparse
import time
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.models.cf_recommender import CollaborativeFilteringRecommender
from src.models.nlp_recommender import NLPRecommender
from src.models.hybrid_recommender import HybridRecommender
from src.evaluation.cold_start_evaluator import ColdStartEvaluator

# Suppress warnings
warnings.filterwarnings('ignore')


def print_section(title: str, char: str = '=', width: int = 70):
    """Print a formatted section header."""
    print('\n' + char * width)
    print(title.center(width))
    print(char * width)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Segmented evaluation (cold/warm/active users)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--users',
        type=int,
        default=100,
        help='Number of test users to evaluate'
    )

    parser.add_argument(
        '--k',
        type=int,
        nargs='+',
        default=[10],
        help='K values for metrics'
    )

    parser.add_argument(
        '--model',
        choices=['cf', 'nlp', 'hybrid', 'all'],
        default='all',
        help='Which model(s) to evaluate'
    )

    parser.add_argument(
        '--cold-threshold',
        type=int,
        default=5,
        help='Maximum interactions for cold-start'
    )

    parser.add_argument(
        '--warm-threshold',
        type=int,
        default=50,
        help='Maximum interactions for warm-start'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    parser.add_argument(
        '--save-results',
        type=str,
        help='Optional path to save results CSV'
    )

    return parser.parse_args()


def main():
    """Main execution."""
    args = parse_args()
    np.random.seed(args.seed)

    start_time = time.time()

    print_section('SEGMENTED EVALUATION - Cold/Warm/Active Users')

    print(f"\nConfiguration:")
    print(f"  Users to evaluate: {args.users}")
    print(f"  K values: {args.k}")
    print(f"  Models: {args.model}")
    print(f"  Cold threshold: < {args.cold_threshold} interactions")
    print(f"  Warm threshold: {args.cold_threshold}-{args.warm_threshold} interactions")
    print(f"  Active: > {args.warm_threshold} interactions")

    # Load data
    print_section('1. Loading Data', char='-')
    loader = DataLoader()
    users_df, likes_df, _ = loader.load_all()
    print(f"✓ Loaded {len(users_df):,} users, {len(likes_df):,} likes")

    # Train/test split
    print_section('2. Creating Train/Test Split', char='-')
    preprocessor = DataPreprocessor(loader)
    train_likes, test_likes = preprocessor.train_test_split_temporal(likes_df, test_size=0.2)
    print(f"✓ Train: {len(train_likes):,} | Test: {len(test_likes):,}")

    # Build ground truth and training sets
    print("\nBuilding ground truth and training interactions...")
    ground_truth = {}
    for uid, grp in test_likes.groupby('user_id'):
        ground_truth[uid] = set(grp['liked_user_id'].unique())

    train_interactions = {}
    for uid, grp in train_likes.groupby('user_id'):
        train_interactions[uid] = set(grp['liked_user_id'].unique())

    # Sample test users
    test_users = list(ground_truth.keys())
    if len(test_users) > args.users:
        test_users = np.random.choice(test_users, size=args.users, replace=False).tolist()

    print(f"✓ Evaluating on {len(test_users)} users")

    # Train models
    print_section('3. Training Models', char='-')
    models = {}

    if args.model in ['cf', 'all']:
        print("\n[CF] Training Collaborative Filtering...")
        start = time.time()
        cf = CollaborativeFilteringRecommender(n_similar_users=50, min_interactions=5)
        cf.fit(train_likes, users_df)
        print(f"  ✓ Trained in {time.time() - start:.1f}s")
        models['CF'] = cf

    if args.model in ['nlp', 'all']:
        print("\n[NLP] Training NLP Recommender...")
        start = time.time()
        nlp = NLPRecommender(model_name='all-MiniLM-L6-v2', batch_size=64)
        nlp.fit(users_df)
        print(f"  ✓ Trained in {time.time() - start:.1f}s")
        models['NLP'] = nlp

    if args.model in ['hybrid', 'all']:
        print("\n[Hybrid] Training Hybrid Recommender...")
        start = time.time()
        hybrid = HybridRecommender(default_alpha=0.6)
        hybrid.fit(train_likes, users_df)
        print(f"  ✓ Trained in {time.time() - start:.1f}s")
        models['Hybrid'] = hybrid

    # Segmented evaluation
    print_section('4. Segmented Evaluation', char='-')

    evaluator = ColdStartEvaluator(
        segment_thresholds=(args.cold_threshold, args.warm_threshold)
    )

    # Print distribution analysis
    evaluator.print_distribution_report(train_interactions)

    # Run segmented evaluation
    results_df = evaluator.evaluate_by_segment(
        models,
        test_users,
        ground_truth,
        train_interactions,
        k_values=args.k
    )

    # Print report
    evaluator.print_segment_report(results_df, k=args.k[0])

    # Save results if requested
    if args.save_results:
        results_df.to_csv(args.save_results, index=False)
        print(f"\n✓ Results saved to {args.save_results}")

    # Summary
    elapsed = time.time() - start_time
    print_section('COMPLETE')
    print(f"\n✓ Segmented evaluation completed in {elapsed:.1f}s")
    print(f"✓ Evaluated {len(test_users)} users on {len(models)} model(s)")
    print(f"✓ Segments: cold/warm/active based on training interactions")

    # Key insights
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)

    if not results_df.empty:
        for k in args.k:
            print(f"\nBest models by segment (K={k}):")

            for segment in ['cold', 'warm', 'active']:
                segment_data = results_df[results_df['Segment'] == segment]

                if not segment_data.empty:
                    # Find best model for this segment
                    metric = f'NDCG@{k}'
                    best_idx = segment_data[metric].idxmax()
                    best_model = segment_data.loc[best_idx, 'Model']
                    best_value = segment_data.loc[best_idx, metric]

                    print(f"  {segment.capitalize():<8} → {best_model:<8} "
                          f"({metric} = {best_value:.4f})")

    print("\n")


if __name__ == '__main__':
    main()

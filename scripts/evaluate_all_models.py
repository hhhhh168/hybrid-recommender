"""
Comprehensive Model Evaluation Script

This script evaluates and compares all three recommendation models:
- Collaborative Filtering (CF)
- NLP-based Recommender
- Hybrid Recommender (CF + NLP)

It computes standard recommendation metrics, generates comparison reports,
and creates visualizations to analyze model performance.

Usage:
    python scripts/evaluate_all_models.py --k_values 5 10 20 --test_size 0.2
    python scripts/evaluate_all_models.py --max_users 1000 --save_dir results/metrics/

Output:
    - JSON: evaluation_results_{timestamp}.json
    - CSV: model_comparison_{timestamp}.csv
    - PNG: metrics_comparison_{timestamp}.png
    - LOG: evaluation.log
"""

import sys
import argparse
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import warnings
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.models.cf_recommender import CollaborativeFilteringRecommender
from src.models.nlp_recommender import NLPRecommender
from src.models.hybrid_recommender import HybridRecommender
from src.evaluation import RecommenderEvaluator
from src.utils import setup_logger, Config

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set matplotlib style for professional plots
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


def setup_logging(save_dir: Path) -> logging.Logger:
    """
    Set up logging to both console and file.

    Args:
        save_dir: Directory to save log file

    Returns:
        Configured logger instance
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    log_file = save_dir / 'evaluation.log'

    # Create logger
    logger = logging.getLogger('evaluation')
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logging to {log_file}")

    return logger


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Evaluate and compare CF, NLP, and Hybrid recommendation models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--k_values',
        type=int,
        nargs='+',
        default=[5, 10, 20],
        help='K values for Precision@K, Recall@K, NDCG@K metrics'
    )

    parser.add_argument(
        '--max_users',
        type=int,
        default=None,
        help='Maximum number of users to evaluate (for speed testing). None = all users'
    )

    parser.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help='Test set size for train/test split (0.0-1.0)'
    )

    parser.add_argument(
        '--save_dir',
        type=str,
        default='results/metrics',
        help='Directory to save evaluation results'
    )

    parser.add_argument(
        '--random_seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.test_size <= 0 or args.test_size >= 1:
        parser.error('test_size must be between 0 and 1')

    if args.max_users is not None and args.max_users <= 0:
        parser.error('max_users must be positive')

    if not args.k_values:
        parser.error('At least one K value must be specified')

    return args


def print_banner(text: str, char: str = '=', width: int = 80) -> None:
    """
    Print a centered banner.

    Args:
        text: Text to display
        char: Character for border
        width: Total width of banner
    """
    print('\n' + char * width)
    print(text.center(width))
    print(char * width)


def print_summary_table(metrics_df: pd.DataFrame, logger: logging.Logger) -> None:
    """
    Print formatted summary table of metrics.

    Args:
        metrics_df: DataFrame with metrics (rows=metrics, cols=models)
        logger: Logger instance
    """
    # Select key metrics for summary
    key_metrics = ['Precision@10', 'Recall@10', 'NDCG@10', 'MAP', 'Coverage', 'Diversity']
    available_metrics = [m for m in key_metrics if m in metrics_df.index]

    if not available_metrics:
        logger.warning("No metrics available for summary table")
        return

    summary_df = metrics_df.loc[available_metrics]

    # Format values as percentages or decimals
    formatted_df = summary_df.copy()
    for col in formatted_df.columns:
        formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.3f}")

    print('\n' + '='*80)
    print('RESULTS SUMMARY'.center(80))
    print('='*80)
    print(formatted_df.to_string())
    print('='*80)


def find_best_model(metrics_df: pd.DataFrame, logger: logging.Logger) -> None:
    """
    Identify and print the best performing model for each metric.

    Args:
        metrics_df: DataFrame with metrics
        logger: Logger instance
    """
    print('\n' + '='*80)
    print('BEST MODELS BY METRIC'.center(80))
    print('='*80)

    for metric in metrics_df.index:
        values = metrics_df.loc[metric]
        best_model = values.idxmax()
        best_value = values.max()
        second_best_value = values.nlargest(2).iloc[-1] if len(values) > 1 else 0

        improvement = ((best_value - second_best_value) / (second_best_value + 1e-9)) * 100

        print(f"{metric:20s} -> {best_model:20s} ({best_value:.4f}, +{improvement:.1f}%)")

    print('='*80)


def create_comparison_visualization(
    metrics_df: pd.DataFrame,
    save_path: Path,
    k_values: List[int],
    logger: logging.Logger
) -> None:
    """
    Create comprehensive visualization comparing all models.

    Args:
        metrics_df: DataFrame with metrics (rows=metrics, cols=models)
        save_path: Path to save the plot
        k_values: List of K values used
        logger: Logger instance
    """
    logger.info("Creating visualization...")

    # Define metric groups
    metric_groups = {
        f'Precision@K (K={k_values})': [f'Precision@{k}' for k in k_values],
        f'Recall@K (K={k_values})': [f'Recall@{k}' for k in k_values],
        f'NDCG@K (K={k_values})': [f'NDCG@{k}' for k in k_values],
        'Overall Metrics': ['MAP', 'Coverage', 'Diversity']
    }

    # Filter available metrics
    available_groups = {}
    for group_name, metrics in metric_groups.items():
        available = [m for m in metrics if m in metrics_df.index]
        if available:
            available_groups[group_name] = available

    n_groups = len(available_groups)
    if n_groups == 0:
        logger.warning("No metrics available for visualization")
        return

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # Color palette
    colors = sns.color_palette('Set2', n_colors=len(metrics_df.columns))
    model_colors = dict(zip(metrics_df.columns, colors))

    for idx, (group_name, metrics) in enumerate(available_groups.items()):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # Get data for this group
        data = metrics_df.loc[metrics].T  # Transpose for easier plotting

        # Create grouped bar chart
        x = np.arange(len(metrics))
        width = 0.8 / len(data)

        for i, (model_name, row) in enumerate(data.iterrows()):
            values = row.values
            offset = (i - len(data) / 2) * width + width / 2
            bars = ax.bar(
                x + offset,
                values,
                width,
                label=model_name,
                color=model_colors[model_name],
                alpha=0.8,
                edgecolor='black',
                linewidth=0.5
            )

            # Add value labels on bars if space permits
            for bar in bars:
                height = bar.get_height()
                if height > 0.02:  # Only show label if bar is tall enough
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.,
                        height,
                        f'{height:.3f}',
                        ha='center',
                        va='bottom',
                        fontsize=7,
                        rotation=0
                    )

        ax.set_xlabel('Metric', fontsize=11, fontweight='bold')
        ax.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax.set_title(group_name, fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right', fontsize=9)
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, min(1.0, data.max().max() * 1.15))

        # Add horizontal line at 0.5 for reference
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)

    # Hide unused subplots
    for idx in range(n_groups, len(axes)):
        axes[idx].axis('off')

    # Overall title
    fig.suptitle(
        'Model Comparison - Hybrid Recommender System',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Visualization saved to {save_path}")


def save_results(
    metrics_df: pd.DataFrame,
    metadata: Dict[str, Any],
    save_dir: Path,
    timestamp: str,
    logger: logging.Logger
) -> Dict[str, Path]:
    """
    Save evaluation results in multiple formats.

    Args:
        metrics_df: DataFrame with metrics
        metadata: Evaluation metadata
        save_dir: Directory to save results
        timestamp: Timestamp string for filenames
        logger: Logger instance

    Returns:
        Dictionary mapping format to file path
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    saved_files = {}

    # 1. Save JSON
    json_path = save_dir / f'evaluation_results_{timestamp}.json'
    results_dict = {
        'evaluation_metadata': metadata,
        'models': metrics_df.to_dict()
    }

    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)

    saved_files['json'] = json_path
    logger.info(f"Results saved to {json_path}")

    # 2. Save CSV
    csv_path = save_dir / f'model_comparison_{timestamp}.csv'
    metrics_df.to_csv(csv_path)
    saved_files['csv'] = csv_path
    logger.info(f"Comparison table saved to {csv_path}")

    return saved_files


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()

    # Set random seed
    np.random.seed(args.random_seed)

    # Setup
    save_dir = Path(args.save_dir)
    logger = setup_logging(save_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Print header
    print_banner('Hybrid Recommender System Evaluation')

    logger.info(f"Configuration:")
    logger.info(f"  K values: {args.k_values}")
    logger.info(f"  Test size: {args.test_size}")
    logger.info(f"  Max users: {args.max_users if args.max_users else 'All'}")
    logger.info(f"  Save directory: {save_dir}")
    logger.info(f"  Random seed: {args.random_seed}")

    try:
        # Step 1: Load data
        logger.info("\n" + "="*80)
        logger.info("STEP 1: Loading data...")
        logger.info("="*80)

        data_loader = DataLoader()
        users_df, likes_df, matches_df = data_loader.load_all()

        logger.info(f"Loaded {len(users_df):,} users, {len(likes_df):,} likes, {len(matches_df):,} matches")

        # Step 2: Create train/test split
        logger.info("\n" + "="*80)
        logger.info(f"STEP 2: Creating train/test split (test_size={args.test_size})...")
        logger.info("="*80)

        preprocessor = DataPreprocessor(data_loader)
        train_likes, test_likes = preprocessor.train_test_split_temporal(
            likes_df,
            test_size=args.test_size
        )

        logger.info(f"Train: {len(train_likes):,} likes | Test: {len(test_likes):,} likes")

        # Step 3: Train models
        logger.info("\n" + "="*80)
        logger.info("STEP 3: Training models...")
        logger.info("="*80)

        models = {}

        # 3a. Collaborative Filtering
        logger.info("\nTraining Collaborative Filtering model...")
        start_time = time.time()
        cf_model = CollaborativeFilteringRecommender(
            n_similar_users=50,
            min_interactions=5,
            like_weight=2.0,
            superlike_weight=3.0
        )
        cf_model.fit(train_likes, users_df)
        cf_time = time.time() - start_time
        models['CF'] = cf_model
        logger.info(f"CF model trained in {cf_time:.1f}s")

        # 3b. NLP Recommender
        logger.info("\nTraining NLP model...")
        start_time = time.time()
        nlp_model = NLPRecommender(
            model_name='all-MiniLM-L6-v2',
            batch_size=32,
            device=None
        )
        nlp_model.fit(users_df)
        nlp_time = time.time() - start_time
        models['NLP'] = nlp_model
        logger.info(f"NLP model trained in {nlp_time:.1f}s")

        # 3c. Hybrid Recommender
        logger.info("\nTraining Hybrid model...")
        start_time = time.time()
        hybrid_model = HybridRecommender(
            cf_model=None,
            nlp_model=None,
            default_alpha=0.6
        )
        hybrid_model.fit(train_likes, users_df)
        hybrid_time = time.time() - start_time
        models['Hybrid'] = hybrid_model
        logger.info(f"Hybrid model trained in {hybrid_time:.1f}s")

        # Step 4: Evaluate models
        logger.info("\n" + "="*80)
        logger.info("STEP 4: Evaluating models...")
        logger.info("="*80)

        # CRITICAL FIX: Pass training data to evaluator for proper exclusion
        evaluator = RecommenderEvaluator(
            models_dict=models,
            test_data=test_likes,
            users_df=users_df,
            train_data=train_likes  # NEW! Ensures proper train/test separation
        )

        logger.info("✓ Evaluator will exclude training interactions from recommendations")

        metrics_df = evaluator.evaluate_all_models(
            k_values=args.k_values,
            max_users=args.max_users
        )

        # Determine actual number of users evaluated
        test_users_evaluated = min(
            len(evaluator.ground_truth),
            args.max_users if args.max_users else len(evaluator.ground_truth)
        )

        logger.info(f"Evaluation complete ({test_users_evaluated} users evaluated)")

        # Step 5: Generate outputs
        logger.info("\n" + "="*80)
        logger.info("STEP 5: Generating outputs...")
        logger.info("="*80)

        # Create metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'test_users': test_users_evaluated,
            'total_test_likes': len(test_likes),
            'total_train_likes': len(train_likes),
            'k_values': args.k_values,
            'test_size': args.test_size,
            'random_seed': args.random_seed,
            'training_times': {
                'CF': f"{cf_time:.2f}s",
                'NLP': f"{nlp_time:.2f}s",
                'Hybrid': f"{hybrid_time:.2f}s"
            }
        }

        # Save results
        saved_files = save_results(metrics_df, metadata, save_dir, timestamp, logger)

        # Create visualization
        viz_path = save_dir / f'metrics_comparison_{timestamp}.png'
        create_comparison_visualization(metrics_df, viz_path, args.k_values, logger)
        saved_files['png'] = viz_path

        # Print summary
        print_summary_table(metrics_df, logger)
        find_best_model(metrics_df, logger)

        # Print file locations
        print('\n' + '='*80)
        print('FILES SAVED'.center(80))
        print('='*80)
        for fmt, path in saved_files.items():
            print(f"  {fmt.upper():6s} -> {path}")
        print(f"  LOG    -> {save_dir / 'evaluation.log'}")
        print('='*80)

        logger.info("\nEvaluation completed successfully!")

    except FileNotFoundError as e:
        logger.error(f"\nData files not found: {e}")
        logger.error("Please ensure data is generated in data/raw/ directory")
        sys.exit(1)

    except Exception as e:
        logger.error(f"\nError during evaluation: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

# Hybrid Recommender System

A hybrid recommendation system for a dating app for white-collar professionals. This system combines collaborative filtering and natural language processing to generate personalized match recommendations.

## 🚀 Modern Tooling

This project uses modern development tools for faster workflows:
- **[Just](https://just.systems)** - Command runner (like Make, but better)
- **[UV](https://docs.astral.sh/uv/)** - Lightning-fast Python package manager (10-100x faster than pip)
- **[Commitizen](https://commitizen-tools.github.io/commitizen/)** - Conventional commits and automated versioning

📖 **See [TOOLING.md](TOOLING.md) for detailed installation and usage guide.**

## Overview

This project implements a production-ready recommendation engine that:
- Analyzes user interaction patterns (hearts, swipes, matches)
- Processes profile text data (bios, job titles, education)
- Combines multiple signals into a unified recommendation score
- Simulates GCP/Firestore functionality using Python ML libraries

## Tech Stack

- **Python 3.9+**
- **NumPy & Pandas**: Data manipulation and matrix operations
- **Scikit-learn**: Machine learning algorithms and metrics
- **Sentence-Transformers**: Text embedding and NLP
- **SciPy**: Sparse matrix operations and similarity calculations
- **PyGeohash**: Location-based features
- **Pytest**: Testing framework
- **Jupyter**: Exploratory data analysis
- **Matplotlib & Seaborn**: Visualization

## Project Structure

```
hybrid-recommender/
├── data/
│   ├── raw/              # Raw CSV/JSON data (gitignored)
│   ├── processed/        # Preprocessed datasets
│   └── generate_data.py  # Synthetic data generation with temporal consistency
├── src/
│   ├── __init__.py
│   ├── utils.py          # Config and logging utilities
│   ├── data_loader.py    # Data loading and preprocessing
│   ├── preprocessing.py  # Feature engineering and data transformation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_recommender.py      # Abstract base class for all models
│   │   ├── cf_recommender.py        # Collaborative filtering model
│   │   ├── nlp_recommender.py       # NLP-based content model
│   │   └── hybrid_recommender.py    # Hybrid model combining CF + NLP
│   └── evaluation/       # Evaluation modules
│       ├── __init__.py
│       ├── recommender_evaluator.py # Core evaluation metrics (RecommenderEvaluator)
│       ├── batch_evaluator.py       # Batch evaluation with sampling
│       └── cold_start_evaluator.py  # Segment-specific evaluation
├── scripts/
│   ├── evaluate_all_models.py  # Comprehensive model comparison
│   └── evaluate_by_segment.py  # Segment-specific evaluation
├── diagnostics/          # Diagnostic and debugging tools
├── notebooks/            # Jupyter notebooks for EDA
├── tests/                # Unit tests
├── results/
│   ├── models/           # Saved models (gitignored)
│   └── metrics/          # Evaluation metrics (gitignored)
├── evaluate_1k_users.py  # Quick evaluation script
├── evaluate_20k_users.py # Full-scale evaluation script
├── justfile              # Just command runner recipes
├── pyproject.toml        # Project metadata and tool configuration
├── requirements.txt      # Python dependencies
├── TOOLING.md            # Modern tooling guide
├── PROJECT_GUIDELINES.md # Development guidelines
└── README.md
```

## Setup

### Prerequisites

Install modern tooling:
```bash
# Install Just (command runner)
brew install just  # macOS
# or: curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash

# Install UV (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Quick Setup (Recommended)

```bash
# 1. Clone the repository
git clone <repository-url>
cd hybrid-recommender

# 2. Setup virtual environment and install dependencies
just setup
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
just install-dev

# 3. View available commands
just
```

### Traditional Setup (Alternative)

```bash
# 1. Clone the repository
git clone <repository-url>
cd hybrid-recommender

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify setup
python -c "from src.utils import Config; print(Config())"
```

## Usage

### Using Just (Recommended)

```bash
# View all available commands
just

# Common workflows
just generate-data      # Generate synthetic data
just eval-quick         # Run evaluation (1k users)
just eval-full          # Run evaluation (20k users)
just test              # Run tests
just clean             # Clean generated files
just dev-cycle         # Clean → Generate → Eval
```

### Manual Commands

```bash
# Generate data
python data/generate_data.py

# Run evaluation
python evaluate_1k_users.py
python evaluate_20k_users.py
python scripts/evaluate_all_models.py

# Run tests
pytest tests/
```

## Architecture

### System Components

The hybrid recommender system consists of three main recommendation models:

#### 1. **Collaborative Filtering (CF) Model**
- **Location**: `src/models/cf_recommender.py`
- **Approach**: User-user and item-item collaborative filtering using sparse matrix operations
- **Features**:
  - Cosine similarity for user/item relationships
  - Efficient sparse matrix computations with SciPy
  - Handles cold-start with fallback strategies
- **Best for**: Finding users with similar interaction patterns

#### 2. **NLP Content Model**
- **Location**: `src/models/nlp_recommender.py`
- **Approach**: Semantic similarity using sentence transformers
- **Features**:
  - Text embeddings from profile bios, job titles, education
  - Pre-trained transformer models (sentence-transformers)
  - Content-based filtering for new users
- **Best for**: Matching based on profile content and interests

#### 3. **Hybrid Model**
- **Location**: `src/models/hybrid_recommender.py`
- **Approach**: Weighted combination of CF and NLP signals
- **Features**:
  - Configurable weights for CF vs NLP scores
  - Combines behavioral and content signals
  - Best overall performance (see evaluation results)
- **Best for**: Production recommendations with balanced signals

### Data Flow

```
1. Data Generation (data/generate_data.py)
   ↓
2. Data Loading (src/data_loader.py)
   ↓
3. Preprocessing (src/preprocessing.py)
   ↓
4. Model Training
   ├── CF Model (collaborative filtering)
   ├── NLP Model (content-based)
   └── Hybrid Model (combined)
   ↓
5. Recommendation Generation
   ↓
6. Evaluation (src/evaluation/)
```

### Key Features

- **Temporal Consistency**: Data generation respects chronological ordering for realistic train/test splits
- **Scalability**: Vectorized operations and sparse matrices for efficient computation
- **Reproducibility**: Fixed random seeds and comprehensive logging
- **Comprehensive Metrics**: Precision@K, Recall@K, NDCG@K, MAP, Coverage, Diversity

## Running Evaluation

The project includes a comprehensive evaluation script that compares all three recommendation models (Collaborative Filtering, NLP, and Hybrid) across multiple metrics.

### Quick Start

Run evaluation with default settings:
```bash
python scripts/evaluate_all_models.py
```

This will:
- Load all data from `data/raw/`
- Create an 80/20 temporal train/test split
- Train all three models (CF, NLP, Hybrid)
- Evaluate using k=[5, 10, 20] for ranking metrics
- Save results to `results/metrics/`

### Command-Line Options

```bash
python scripts/evaluate_all_models.py \
    --k_values 5 10 20 \
    --test_size 0.2 \
    --max_users 1000 \
    --save_dir results/metrics/ \
    --random_seed 42
```

**Arguments:**
- `--k_values`: K values for Precision@K, Recall@K, NDCG@K (default: 5 10 20)
- `--test_size`: Train/test split ratio (default: 0.2)
- `--max_users`: Max users to evaluate, for faster testing (default: None = all users)
- `--save_dir`: Output directory for results (default: results/metrics/)
- `--random_seed`: Random seed for reproducibility (default: 42)

### Example Runs

Evaluate on a sample of 500 users (faster):
```bash
python scripts/evaluate_all_models.py --max_users 500
```

Evaluate with different K values:
```bash
python scripts/evaluate_all_models.py --k_values 3 5 10 15 20
```

Save results to custom directory:
```bash
python scripts/evaluate_all_models.py --save_dir experiments/run_001/
```

### Output Files

The script generates four output files with timestamps:

1. **evaluation_results_{timestamp}.json**
   ```json
   {
     "evaluation_metadata": {
       "timestamp": "2025-10-11T14:30:22",
       "test_users": 1000,
       "k_values": [5, 10, 20],
       "training_times": {...}
     },
     "models": {
       "CF": {"Precision@5": 0.234, ...},
       "NLP": {"Precision@5": 0.143, ...},
       "Hybrid": {"Precision@5": 0.218, ...}
     }
   }
   ```

2. **model_comparison_{timestamp}.csv**
   - Comparison table with models as columns and metrics as rows
   - Easy to import into Excel or visualization tools

3. **metrics_comparison_{timestamp}.png**
   - Grouped bar charts comparing models across all metrics
   - High-resolution (300 DPI) for presentations

4. **evaluation.log**
   - Detailed logs of the entire evaluation process
   - Useful for debugging and understanding model behavior

### Metrics Computed

The evaluation computes the following metrics:

**Ranking Metrics** (for each K):
- **Precision@K**: Fraction of top-K recommendations that are relevant
- **Recall@K**: Fraction of relevant items found in top-K
- **NDCG@K**: Normalized Discounted Cumulative Gain (position-aware)

**Overall Metrics**:
- **MAP**: Mean Average Precision across all users
- **Coverage**: Fraction of catalog items recommended to at least one user
- **Diversity**: Average intra-list diversity (1 - pairwise similarity)

### Expected Output

```
========================================
Hybrid Recommender System Evaluation
========================================

Loading data...
✓ Loaded 20,000 users, 659,086 likes, 1,424 matches

Creating train/test split (test_size=0.2)...
✓ Train: 527,269 likes | Test: 131,817 likes

Training Collaborative Filtering model...
✓ CF model trained in 8.3s

Training NLP model...
✓ NLP model trained in 45.2s

Training Hybrid model...
✓ Hybrid model trained in 52.1s

Evaluating models...
✓ Evaluation complete (1000 users evaluated)

========================================
           RESULTS SUMMARY
========================================
                      CF      NLP    Hybrid
Precision@10       0.187    0.143     0.218
Recall@10          0.234    0.198     0.287
NDCG@10            0.289    0.221     0.334
MAP                0.156    0.128     0.192
Coverage           0.452    0.378     0.501
Diversity          0.634    0.712     0.689
========================================

Best Model: Hybrid (16.6% better Precision@10 than CF)

Results saved to:
  - results/metrics/evaluation_results_20251011_143022.json
  - results/metrics/model_comparison_20251011_143022.csv
  - results/metrics/metrics_comparison_20251011_143022.png
```

### Troubleshooting

**Missing data files:**
```
FileNotFoundError: Users file not found: data/raw/users.csv
```
**Solution**: Ensure you've generated data first. Check that `data/raw/` contains `users.csv`, `likes.csv`, and `matches.csv`.

**Out of memory:**
```
MemoryError: Unable to allocate array
```
**Solution**: Use `--max_users` to evaluate on a smaller sample:
```bash
python scripts/evaluate_all_models.py --max_users 500
```

**Slow evaluation:**
**Solution**: The NLP model requires downloading embeddings on first run. Subsequent runs will be faster. You can also reduce the number of test users with `--max_users`.

## Development

### Making Commits

This project uses **Commitizen** for conventional commits:

```bash
# Stage your changes
git add .

# Create a conventional commit (interactive prompt)
just commit
# or: cz commit

# Push changes
git push
```

### Available Documentation

- **[TOOLING.md](TOOLING.md)** - Modern tooling guide (Just, UV, Commitizen)
- **[PROJECT_GUIDELINES.md](PROJECT_GUIDELINES.md)** - Development guidelines, code style, testing
- **[justfile](justfile)** - All available commands and workflows

## License

_(To be added)_

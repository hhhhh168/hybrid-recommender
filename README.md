# Hybrid Recommender System

A hybrid recommendation system for a dating app for white-collar professionals. This system combines collaborative filtering and natural language processing to generate personalized match recommendations.

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
│   └── processed/        # Preprocessed datasets
├── src/
│   ├── __init__.py
│   ├── utils.py          # Config and logging utilities
│   ├── data_generator.py # (To be added)
│   ├── collaborative.py  # (To be added)
│   ├── nlp.py           # (To be added)
│   └── hybrid.py        # (To be added)
├── notebooks/            # Jupyter notebooks for EDA
├── tests/               # Unit tests
├── results/
│   ├── models/          # Saved models (gitignored)
│   └── metrics/         # Evaluation metrics (gitignored)
├── requirements.txt
├── CLAUDE.md           # Development guidelines
└── README.md
```

## Setup

### 1. Clone the repository
```bash
git clone <repository-url>
cd hybrid-recommender
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify setup
```bash
python -c "from src.utils import Config; print(Config())"
```

## Usage

_(To be added: Instructions for data generation, model training, and generating recommendations)_

## Architecture

_(To be added: System architecture diagram and component descriptions)_

## Evaluation

_(To be added: Performance metrics and evaluation results)_

## Development

See [CLAUDE.md](CLAUDE.md) for detailed development guidelines including:
- Code style requirements
- Firestore schema reference
- Commit conventions
- Testing guidelines

## License

_(To be added)_

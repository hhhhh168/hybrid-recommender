# Hybrid Recommender System - Justfile
# Modern command runner for common project tasks
# https://github.com/casey/just

# Default recipe to display help information
default:
    @just --list

# Install dependencies using UV (fast Python package manager)
install:
    uv pip install -r requirements.txt

# Create and setup a virtual environment using UV
setup:
    uv venv
    @echo "Virtual environment created. Activate with: source .venv/bin/activate"

# Run all tests with pytest
test:
    pytest tests/ -v

# Run tests with coverage report
test-coverage:
    pytest tests/ --cov=src --cov-report=html --cov-report=term

# Generate synthetic data for the recommender system
generate-data:
    python src/data_generator.py

# Run full evaluation on all models (1k users for quick testing)
eval-quick:
    python evaluate_1k_users.py

# Run full evaluation on all models (20k users - production scale)
eval-full:
    python evaluate_20k_users.py

# Run evaluation with custom parameters
eval k_values="5 10 20" test_size="0.2" max_users="1000":
    python scripts/evaluate_all_models.py --k_values {{k_values}} --test_size {{test_size}} --max_users {{max_users}}

# Clean generated data and results
clean:
    rm -rf data/raw/*.csv
    rm -rf data/processed/*.pkl
    rm -rf results/models/*
    rm -rf results/metrics/*
    rm -rf __pycache__
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete

# Clean and regenerate everything
reset: clean generate-data

# Format code (when ruff is added)
fmt:
    @echo "Install ruff with: uv pip install ruff"
    @echo "Then run: ruff format ."

# Lint code (when ruff is added)
lint:
    @echo "Install ruff with: uv pip install ruff"
    @echo "Then run: ruff check ."

# Run linter and tests
check: lint test

# Start Jupyter notebook server
notebook:
    jupyter notebook notebooks/

# Create a conventional commit using Commitizen
commit:
    cz commit

# Bump version and create changelog using Commitizen
bump:
    cz bump --changelog

# Show project status
status:
    @echo "📊 Hybrid Recommender System Status"
    @echo "===================================="
    @echo ""
    @echo "📁 Data Files:"
    @ls -lh data/raw/*.csv 2>/dev/null || echo "  No data files found (run: just generate-data)"
    @echo ""
    @echo "🧪 Test Files:"
    @find tests -name "test_*.py" -type f | wc -l | xargs echo "  Test files:"
    @echo ""
    @echo "📈 Recent Evaluations:"
    @ls -lt results/metrics/*.json 2>/dev/null | head -3 || echo "  No evaluations found"

# Install development dependencies
install-dev:
    uv pip install -r requirements.txt
    uv pip install pytest pytest-cov ruff commitizen

# Quick development cycle: clean, generate data, run quick eval
dev-cycle: clean generate-data eval-quick
    @echo "✅ Development cycle complete!"

# Production-ready workflow: clean, generate data, test, full eval
prod-cycle: clean generate-data test eval-full
    @echo "✅ Production cycle complete!"

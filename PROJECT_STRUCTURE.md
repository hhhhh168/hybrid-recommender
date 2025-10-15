# Project Structure

## Directory Organization

```
hybrid-recommender/
├── src/                         # Core source code
│   ├── models/                  # Recommendation models (CF, NLP, Hybrid)
│   ├── evaluation/              # Evaluation framework
│   ├── data_loader.py           # Data loading utilities
│   ├── preprocessing.py         # Data preprocessing
│   └── utils.py                 # Common utilities
│
├── scripts/                     # Production scripts
│   ├── evaluate_all_models.py   # Main comprehensive evaluation
│   ├── evaluate_by_segment.py   # Segmented evaluation (cold/warm/active)
│   └── (future: generate_data.py)
│
├── diagnostics/                 # Debugging & diagnostic tools
│   ├── diagnose_eval.py         # Evaluate exclusion logic
│   ├── diagnose_model_quality.py # Check model predictions
│   └── validate_bio_diversity.py # Data quality checks
│
├── tests/                       # Unit tests (pytest)
│   ├── test_cf_model.py         # CF model tests
│   ├── test_models.py           # General model tests
│   └── test_pipeline.py         # Pipeline tests
│
├── data/                        # Data files
│   ├── raw/                     # Generated data (CSV files)
│   ├── processed/               # Preprocessed data
│   ├── generate_data.py         # Data generation script
│   └── validate_bios.py         # Bio validation
│
├── results/                     # Evaluation results (git ignored)
│   └── metrics/                 # Metric outputs
│
├── evaluate_1k_users.py         # Quick evaluation (kept in root for convenience)
│
└── Documentation files (*.md)
    ├── README.md                         # Main project documentation
    ├── PROJECT_GUIDELINES.md             # Development guidelines
    ├── EVALUATION_DIAGNOSIS_REPORT.md    # Evaluation system diagnosis
    ├── DATA_GENERATION_IMPROVEMENTS.md   # Data generation improvements
    ├── CLEANUP_PLAN.md                   # Script cleanup plan
    └── PROJECT_STRUCTURE.md              # This file
```

## Quick Reference

### Run Evaluations
```bash
# Quick evaluation (1K users, ~10-15 min)
python evaluate_1k_users.py

# Full evaluation (all models, all users)
python scripts/evaluate_all_models.py

# Segmented evaluation (by user activity)
python scripts/evaluate_by_segment.py --users 500
```

### Diagnostics
```bash
# Check evaluation logic
python diagnostics/diagnose_eval.py

# Check model quality
python diagnostics/diagnose_model_quality.py

# Validate bio diversity
python diagnostics/validate_bio_diversity.py
```

### Generate Data
```bash
# Generate synthetic data
python data/generate_data.py
```

### Run Tests
```bash
# Run all unit tests
pytest tests/

# Run specific test
pytest tests/test_cf_model.py
```

## Cleanup Summary

**Removed (6 redundant scripts):**
- `evaluate_with_new_data.py`
- `manual_eval.py`
- `test_batch_evaluator.py`
- `test_batch_methods.py`
- `test_reciprocal_generation.py`
- `test_vectorized_eval.py`

**Reorganized:**
- Diagnostic scripts → `diagnostics/`
- Unit tests → `tests/`
- Evaluation scripts → `scripts/` (except quick eval in root)

## Best Practices

1. **Keep root directory clean** - Only essential entry points
2. **Organize by purpose** - Production scripts vs diagnostics vs tests
3. **Clear naming** - descriptive filenames, avoid generic "test" prefix
4. **Documentation** - Each directory has clear purpose
5. **Git ignore** - Results and data/raw/ not committed

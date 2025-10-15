# Script Cleanup Plan

## Analysis of Redundant Scripts

### Diagnostic Scripts (Keep Useful Ones)
- **diagnose_eval.py** (5.7K) - ✅ **KEEP** - Useful for debugging evaluation logic
- **diagnose_model_quality.py** (4.5K) - ✅ **KEEP** - Useful for checking model predictions
- **test_vectorized_eval.py** (3.0K) - ❌ **REMOVE** - Superseded by evaluate_1k_users.py

### Evaluation Scripts (Consolidate)
- **scripts/evaluate_all_models.py** - ✅ **KEEP** - Main comprehensive evaluation
- **evaluate_1k_users.py** (7.7K) - ✅ **KEEP** - Quick evaluation with batching
- **evaluate_by_segment.py** (7.3K) - ⚠️ **REVIEW** - May be redundant
- **evaluate_with_new_data.py** (4.2K) - ❌ **REMOVE** - Redundant
- **manual_eval.py** (11K) - ❌ **REMOVE** - Old/ad-hoc testing

### Test Scripts (Cleanup Old Tests)
- **test_batch_evaluator.py** (5.3K) - ❌ **REMOVE** - Testing old code
- **test_batch_methods.py** (3.0K) - ❌ **REMOVE** - Testing old code
- **test_bio_diversity.py** (18K) - ⚠️ **ARCHIVE** - Data quality check, move to scripts/
- **test_cf_model.py** (4.9K) - ⚠️ **MOVE TO tests/** - Unit test
- **test_models.py** (8.4K) - ⚠️ **MOVE TO tests/** - Unit test
- **test_pipeline.py** (4.5K) - ⚠️ **MOVE TO tests/** - Unit test
- **test_reciprocal_generation.py** (2.7K) - ❌ **REMOVE** - Old data generation test

## Recommended Structure

```
hybrid-recommender/
├── scripts/                    # Production scripts
│   ├── evaluate_all_models.py  # Main evaluation
│   ├── generate_data.py        # Data generation (move from data/)
│   └── validate_data.py        # Data quality checks
├── tests/                      # Unit tests (pytest)
│   ├── test_cf_model.py
│   ├── test_models.py
│   └── test_pipeline.py
├── diagnostics/                # Debugging tools (NEW)
│   ├── diagnose_eval.py
│   ├── diagnose_model_quality.py
│   └── validate_bio_diversity.py
├── evaluate_1k_users.py        # Keep in root for quick access
└── [remove all other test_*.py, evaluate_*.py]
```

## Actions

### 1. Remove Redundant Scripts
```bash
rm evaluate_with_new_data.py
rm manual_eval.py
rm test_batch_evaluator.py
rm test_batch_methods.py
rm test_reciprocal_generation.py
rm test_vectorized_eval.py
```

### 2. Create New Directories
```bash
mkdir -p diagnostics
mkdir -p tests
```

### 3. Move Scripts to Proper Locations
```bash
# Move diagnostics
mv diagnose_eval.py diagnostics/
mv diagnose_model_quality.py diagnostics/
mv test_bio_diversity.py diagnostics/validate_bio_diversity.py

# Move unit tests
mv test_cf_model.py tests/
mv test_models.py tests/
mv test_pipeline.py tests/

# Move data generation to scripts
mv data/generate_data.py scripts/generate_data.py
```

### 4. Decision on evaluate_by_segment.py
**Review needed** - Check if it provides unique value over evaluate_all_models.py
- If YES: Move to scripts/
- If NO: Remove

## Final Structure

After cleanup, root directory will only have:
- `evaluate_1k_users.py` - Quick evaluation entry point
- All other scripts organized in subdirectories
- Clean, professional project structure

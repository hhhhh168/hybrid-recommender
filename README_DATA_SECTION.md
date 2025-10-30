# README Data Section - COPY THIS INTO YOUR README.md

Add this section after "## Overview" and before "## Tech Stack" in your README.md

---

## Data

**Data Source**: Synthetic data generated for this portfolio project.

Real dating app data cannot be publicly shared due to privacy concerns (sensitive PII). This project uses carefully crafted synthetic data that mimics production-level characteristics.

### Dataset Statistics
- **Users**: 20,000 (white-collar professionals, ages 25-45)
- **Interactions**: ~2,400,000 likes (~120 per user)
- **Matches**: ~50,000 (2% match rate, realistic for dating apps)
- **Timeline**: 1-year operation period
- **Matrix Sparsity**: 99.4% (comparable to Netflix Prize 99%+)

### Data Generation Approach

The synthetic data incorporates realistic patterns to enable meaningful algorithm evaluation:

1. **Power-Law Popularity**: Some users are very popular (many likes), most have average popularity
2. **Collaborative Filtering Signals**: User clustering (5 mega-clusters) creates learnable behavioral patterns
3. **Content-Based Patterns**: Profile text similarity correlates with likes
4. **Temporal Consistency**: User preferences remain stable over time (enables proper train/test splits)
5. **Cold Start Handling**: Realistic mix of highly active and new users

### Why Synthetic?

- **Privacy**: Real dating app data contains sensitive personal information (names, photos, messages)
- **Availability**: No public dating app datasets exist at this scale
- **Control**: Enables validation with known ground-truth patterns
- **Reproducibility**: Anyone can regenerate identical data for verification

### Comparison to Real Systems

| Metric | This Project | Real Systems |
|--------|-------------|--------------|
| **Matrix Sparsity** | 99.4% | Netflix Prize: 99%+, MovieLens: 98-99% |
| **Popularity Distribution** | Power-law (Gini ~0.8) | Social platforms: 0.8-0.9 |
| **Scale** | 20K users, 2.4M interactions | MovieLens 20M: 138K users, 20M ratings |
| **Cold Start** | 20-30% users with <5 interactions | Production typical: 20-30% |
| **Avg Interactions/User** | 120 | Dating apps: 50-200 |

The synthetic data matches production system characteristics, ensuring algorithms are validated against realistic conditions.

See `data/generate_data.py` for detailed implementation.

---

# OPTIONAL: Add this section after running the validation audit

## Data Validation

The synthetic data has been validated against production system characteristics:

| Check | Status | Details |
|-------|--------|---------|
| **Sparsity** | ✅ Pass | 99.4% (target: >95%) |
| **Popularity Distribution** | ✅ Pass | Gini: 0.82 (target: >0.75) |
| **Cold Start** | ✅ Pass | 23% users <5 interactions |
| **Power Law** | ✅ Pass | Long-tail distribution confirmed |
| **Temporal Consistency** | ✅ Pass | User preferences stable over time |

Run validation audit:
```bash
python audit_synthetic_data.py
```

Results available in `validation_results/`:
- `validation_report.json` - Full metrics
- `popularity_distribution.png` - Power-law visualization
- `cold_start_analysis.png` - User/item activity distribution

---

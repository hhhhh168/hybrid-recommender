# Hybrid Recommender System - Development Guidelines

## Project Context

This is a dating app for white-collar professionals. This project implements a hybrid recommendation system that combines:
- Collaborative filtering (user behavior patterns)
- Natural Language Processing (profile text analysis)

The system simulates a production environment using Python ML libraries to mimic GCP/Firestore functionality.

## Code Style

### Python Standards
- Follow **PEP 8** style guide
- Use **type hints** for all function parameters and return values
- Include **docstrings** for all modules, classes, and functions (Google style)
- Maximum line length: 100 characters
- Use meaningful variable names (no single letters except in loops/comprehensions)

### Example:
```python
def calculate_similarity(
    user_id: str,
    candidate_id: str,
    weights: dict[str, float]
) -> float:
    """
    Calculate similarity score between two users.

    Args:
        user_id: ID of the target user
        candidate_id: ID of the candidate user
        weights: Feature weights for similarity calculation

    Returns:
        Similarity score between 0 and 1
    """
    pass
```

## Firestore Schema Fields

The app uses Firestore with the following user profile schema:

### Core Profile Fields:
- `uid` (string): Unique user identifier
- `bio` (string): User biography/description
- `birthday` (timestamp): User's date of birth
- `location` (object): User location data
  - `geohash` (string): Geohashed location
  - `latitude` (number)
  - `longitude` (number)
- `jobTitle` (string): Current job title
- `school` (string): Educational institution
- `gender` (string): User's gender
- `interestedIn` (string): Gender preference

### Engagement Fields:
- `hearts` (array): List of user IDs who hearted this user
- `swipesLeft` (array): List of user IDs who swiped left
- `swipesRight` (array): List of user IDs who swiped right
- `matches` (array): List of matched user IDs

### Premium/Settings:
- `isPremium` (boolean): Premium subscription status
- `ageRange` (object): Preference for age range
  - `min` (number)
  - `max` (number)
- `distancePreference` (number): Maximum distance in miles

### Additional Fields:
- `photos` (array): Photo URLs
- `interests` (array): List of interest tags
- `height` (number): Height in inches
- `education` (string): Education level
- `ethnicity` (string)
- `religion` (string)

## Commit Conventions

Use conventional commits format:
```
<type>(<scope>): <description>

[optional body]
```

### Types:
- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code refactoring
- `docs`: Documentation changes
- `test`: Adding/updating tests
- `chore`: Maintenance tasks

### Examples:
```
feat(collab): implement user-item matrix construction
fix(nlp): correct sentence embedding dimension mismatch
refactor(data): optimize data loading pipeline
test(recommender): add unit tests for hybrid scoring
```

## Important Reminders

### Do NOT Use:
- `localStorage` - This is a Python/ML project, not a web frontend
- Hardcoded file paths - Use the `Config` class from `src/utils.py`
- Magic numbers - Define constants at module level or in config

### Do Use:
- Relative imports within the `src/` package
- The logger from `src/utils.py` for debugging
- Type hints and dataclasses for data structures
- Vectorized operations (NumPy/Pandas) over loops where possible

## Testing

- Write unit tests in `tests/` directory
- Use pytest for testing framework
- Aim for >80% code coverage
- Test edge cases and data validation

## Data Privacy

- Never commit real user data to version control
- Use synthetic/fake data for development (see Faker library)
- Anonymize any example data
- Keep data files in `data/raw/` (gitignored)

## Performance Considerations

- Profile code before optimizing
- Use batch processing for large datasets
- Cache expensive computations where appropriate
- Document time/space complexity for key algorithms

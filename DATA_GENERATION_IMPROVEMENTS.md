# Data Generation Improvements for Learnable Patterns

## Current Problem

The synthetic data generation creates **random, unpredictable like patterns** that make it impossible for ML models to learn meaningful recommendations.

**Evidence:**
- Users' training likes have 0% overlap with test likes
- User behavior changes drastically between time periods
- No collaborative filtering signals (similar users don't like similar people)
- No content-based signals (profile similarity doesn't correlate with likes)

---

## Required Improvements

### 1. **User Preference Vectors (Stable Preferences)**

**Problem**: Currently each like is independent and random
**Solution**: Give each user stable preference vectors

```python
# Current (Random):
user_likes = random.sample(all_users, k=random.randint(50, 150))

# Improved (Preference-Based):
user_preferences = {
    'age_preference': normal(30, 5),
    'location_preference': user['city'],
    'industry_preference': ['Tech', 'Finance'],
    'personality_vector': generate_stable_vector(dim=10)
}

# Like probability based on match to preferences
for candidate in all_users:
    match_score = compute_preference_match(user_preferences, candidate)
    like_probability = sigmoid(match_score)
    if random() < like_probability:
        likes.append(candidate)
```

### 2. **Collaborative Filtering Signals**

**Problem**: Similar users don't exhibit similar behavior
**Solution**: Create user clusters where cluster members like similar people

```python
# Create 5-10 mega-clusters
clusters = create_user_clusters(users, n_clusters=8)

# Users in same cluster should like people from specific target clusters
cluster_affinities = {
    0: [2, 5, 7],  # Cluster 0 likes people from clusters 2, 5, 7
    1: [1, 3, 4],
    # ...
}

# Generate likes based on cluster membership
for user in cluster_0:
    target_clusters = cluster_affinities[0]
    candidates = users_in_clusters(target_clusters)
    user_likes = preference_sample(candidates, user.preferences)
```

### 3. **Content-Based Signals (Profile Similarity)**

**Problem**: Profile text similarity doesn't correlate with likes
**Solution**: Make bio/job/interests influence like probability

```python
# Compute profile similarity
profile_sim = cosine_similarity(user.bio_embedding, candidate.bio_embedding)

# Bio similarity should increase like probability
like_prob_base = 0.1  # base rate
like_prob_with_bio = like_prob_base + (0.3 * profile_sim)  # up to +30%

# Keywords matching increases probability
if any(keyword in candidate.bio for keyword in user.interests):
    like_prob_with_bio += 0.2
```

### 4. **Temporal Consistency**

**Problem**: User behavior changes completely between train/test
**Solution**: Preferences evolve slowly, with high overlap

```python
# Training period (days 0-300)
train_preferences = user.initial_preferences

# Test period (days 301-365)
# Preferences evolve slightly, not completely change
test_preferences = {
    k: v + random.normal(0, 0.1)  # Small random walk
    for k, v in train_preferences.items()
}

# Ensure 30-50% overlap between train and test likes
# by using similar preference vectors
```

### 5. **Realistic Distribution Shapes**

**Problem**: Uniform random distributions
**Solution**: Power-law and realistic distributions

```python
# Number of likes: Power law (most users like 20-50, few like 200+)
n_likes = int(powerlaw.rvs(a=2.5, loc=20, scale=30))

# Who gets liked: Popularity follows power law
# (some users very popular, most average)
popularity_scores = powerlaw.rvs(a=2.0, size=n_users)

# Like probability proportional to popularity + preference match
like_prob = 0.7 * preference_match + 0.3 * popularity_score
```

---

## Specific Implementation Plan

### Phase 1: Add User Preference Vectors (HIGH PRIORITY)

```python
def create_user_preferences(user, seed):
    """Create stable preference vector for user."""
    rng = np.random.RandomState(seed + hash(user['user_id']))

    return {
        # Demographic preferences (stable)
        'age_mean': rng.normal(user['age'], 3),
        'age_std': rng.uniform(3, 8),
        'location_weight': rng.uniform(0.3, 0.8),

        # Content preferences (stable personality)
        'personality_vector': rng.randn(10),  # 10-dim personality
        'industry_affinity': rng.dirichlet([2]*5),  # prefer certain industries

        # Behavioral traits
        'selectivity': rng.beta(2, 5),  # how picky (0=not picky, 1=very picky)
        'reciprocity': rng.uniform(0.6, 0.9),  # prefers mutual interest
    }
```

### Phase 2: Implement Cluster-Based Generation (HIGH PRIORITY)

```python
# Create clusters using K-means on user features
from sklearn.cluster import KMeans

features = np.column_stack([
    users['age'],
    users['city_encoded'],
    users['industry_encoded'],
    bio_embeddings  # from sentence-transformers
])

clusters = KMeans(n_clusters=10, random_state=42).fit_predict(features)

# Define cluster affinities (which clusters like which)
# This creates strong CF signals
cluster_affinity_matrix = create_affinity_matrix(n_clusters=10)
```

### Phase 3: Content-Based Weighting (MEDIUM PRIORITY)

```python
def compute_like_probability(user, candidate, user_prefs):
    """Compute probability user likes candidate."""

    # Demographics match
    age_match = normal_pdf(candidate['age'],
                           user_prefs['age_mean'],
                           user_prefs['age_std'])

    # Location match
    location_match = 1.0 if candidate['city'] == user['city'] else 0.3

    # Bio similarity (NLP)
    bio_sim = cosine_similarity(user['bio_embedding'],
                                 candidate['bio_embedding'])

    # Cluster affinity (CF signal)
    cluster_affinity = affinity_matrix[user['cluster'], candidate['cluster']]

    # Combine (weighted sum)
    score = (
        0.25 * age_match +
        0.20 * location_match +
        0.30 * bio_sim +
        0.25 * cluster_affinity
    )

    # Apply selectivity
    threshold = user_prefs['selectivity']
    like_prob = sigmoid(5 * (score - threshold))

    return like_prob
```

### Phase 4: Temporal Split with Consistency (HIGH PRIORITY)

```python
def generate_likes_with_temporal_consistency(user, candidates,
                                              time_period='train'):
    """Generate likes ensuring train/test consistency."""

    # Get user's stable preferences
    prefs = user['preferences']

    # Compute like probabilities for all candidates
    like_probs = [compute_like_probability(user, c, prefs)
                  for c in candidates]

    # Sample likes based on probabilities
    n_likes_mean = 100 if time_period == 'train' else 25
    n_likes = max(5, int(np.random.normal(n_likes_mean, 20)))

    # Sample using probabilities (ensures overlap!)
    # Same candidates will have high prob in both periods
    liked_users = weighted_sample(candidates, like_probs, n=n_likes)

    return liked_users
```

---

## Expected Outcome After Improvements

### Before (Current):
```
Precision@10: 0.0068 (0.68%)
Recall@10:    0.0049 (0.49%)
NDCG@10:      0.0065 (0.65%)

Train/Test Overlap: 0%
Model Hit Rate: 0/20 (0%)
```

### After (Fixed Data):
```
Precision@10: 0.15-0.25 (15-25%)
Recall@10:    0.10-0.20 (10-20%)
NDCG@10:      0.20-0.35 (20-35%)

Train/Test Overlap: 30-50%
Model Hit Rate: 3-5/20 (15-25%)
```

---

## Validation Checklist

After implementing improvements, verify:

- [ ] Users have stable preference vectors (saved with user data)
- [ ] User clusters exist and are used for generation
- [ ] Like probability is based on multiple factors (not random)
- [ ] Training and test likes show 30-50% overlap per user
- [ ] Similar users (same cluster) like similar people
- [ ] Profile similarity correlates with like probability (r > 0.3)
- [ ] Distribution of likes follows power law, not uniform
- [ ] Temporal consistency: same user preferences in train/test
- [ ] Re-run evaluation shows 15-25% Precision@10

---

## Implementation Priority

1. **🔴 Critical**: Add stable user preferences (fixes temporal consistency)
2. **🔴 Critical**: Implement cluster-based generation (fixes CF signals)
3. **🔴 Critical**: Ensure train/test overlap 30-50% (fixes predictability)
4. **🟡 Important**: Add content-based weighting (improves NLP signals)
5. **🟡 Important**: Power-law distributions (improves realism)
6. **🟢 Nice-to-have**: Advanced features (reciprocity, time decay, etc.)

---

## Testing Strategy

1. **Generate new data** with improvements
2. **Run diagnostics**:
   ```bash
   python diagnose_model_quality.py  # Should show 15-25% hit rate
   ```
3. **Run full evaluation**:
   ```bash
   python evaluate_1k_users.py  # Should show 15-25% Precision@10
   ```
4. **Verify patterns**:
   - Check train/test overlap > 30%
   - Verify cluster members like similar people
   - Confirm profile similarity correlates with likes

"""
Generate realistic synthetic data for dating app recommendation system.

This script generates:
- 20,000 users (white-collar professionals, ages 25-45)
- ~2,400,000 likes (power law distribution over 1-year timeline, 120 avg/user)
- ~50,000 matches (~2% match rate with strong CF patterns)

Data simulates 1 year of app operation with VERY STRONG collaborative filtering patterns:
- 5 MEGA-CLUSTERS (~4000 users each) for extreme concentration
- Multi-signal preference scoring (NLP + demographics + geo + recency)
- 95% preference-driven, 5% exploration
- Minimum 15 likes/user (eliminates cold-start problem)
- Realistic superlike rates (5-10%)
"""

import random
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from faker import Faker
import pygeohash as pgh
from collections import defaultdict
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)
fake = Faker()
Faker.seed(42)

# Configuration
NUM_USERS = 20000
TARGET_TOTAL_LIKES = 2400000  # ~120 avg likes/user to ensure no cold-start
TARGET_MATCHES = 25000
PREMIUM_RATE = 0.15
SUPER_RATE = 0.05
APP_START_DATE = datetime.now() - timedelta(days=365)
CURRENT_DATE = datetime.now()

# Industry distributions
INDUSTRIES = {
    'Technology': {
        'weight': 0.25,
        'titles': [
            'Software Engineer', 'Senior Software Engineer', 'Product Manager',
            'Data Scientist', 'Engineering Manager', 'UX Designer', 'DevOps Engineer',
            'Machine Learning Engineer', 'Technical Program Manager', 'Frontend Developer',
            'Backend Developer', 'Full Stack Engineer', 'UI/UX Designer'
        ]
    },
    'Healthcare': {
        'weight': 0.25,
        'titles': [
            'Physician', 'Nurse Practitioner', 'Healthcare Administrator',
            'Pharmacist', 'Medical Researcher', 'Physician Assistant',
            'Clinical Psychologist', 'Dentist', 'Physical Therapist',
            'Medical Director', 'Clinical Research Coordinator'
        ]
    },
    'Government & Education': {
        'weight': 0.20,
        'titles': [
            'Policy Analyst', 'High School Teacher', 'Professor',
            'Government Official', 'Public Administrator', 'Research Scientist',
            'Education Administrator', 'University Lecturer', 'Curriculum Specialist',
            'School Principal', 'Legislative Aide', 'City Planner'
        ]
    },
    'Finance': {
        'weight': 0.20,
        'titles': [
            'Investment Banker', 'Financial Analyst', 'Accountant',
            'Wealth Manager', 'Trader', 'Portfolio Manager', 'Risk Analyst',
            'VP Finance', 'Corporate Finance Manager', 'CPA', 'Auditor'
        ]
    },
    'Legal': {
        'weight': 0.04,
        'titles': [
            'Attorney', 'Corporate Lawyer', 'Legal Counsel',
            'Public Defender', 'Prosecutor', 'Patent Attorney', 'Paralegal Manager'
        ]
    },
    'Consulting': {
        'weight': 0.03,
        'titles': [
            'Management Consultant', 'Strategy Consultant', 'Business Analyst',
            'Senior Consultant', 'Principal Consultant'
        ]
    },
    'Other': {
        'weight': 0.03,
        'titles': [
            'Architect', 'Marketing Manager', 'Communications Director',
            'Nonprofit Director', 'Real Estate Developer', 'Urban Planner'
        ]
    }
}

# Major US cities with real coordinates
CITIES = {
    'San Francisco': {'lat': 37.7749, 'lng': -122.4194, 'state': 'CA'},
    'New York': {'lat': 40.7128, 'lng': -74.0060, 'state': 'NY'},
    'Washington DC': {'lat': 38.9072, 'lng': -77.0369, 'state': 'DC'},
    'Austin': {'lat': 30.2672, 'lng': -97.7431, 'state': 'TX'},
    'Seattle': {'lat': 47.6062, 'lng': -122.3321, 'state': 'WA'},
    'Boston': {'lat': 42.3601, 'lng': -71.0589, 'state': 'MA'},
    'Los Angeles': {'lat': 34.0522, 'lng': -118.2437, 'state': 'CA'},
    'Chicago': {'lat': 41.8781, 'lng': -87.6298, 'state': 'IL'},
    'Denver': {'lat': 39.7392, 'lng': -104.9903, 'state': 'CO'},
    'Miami': {'lat': 25.7617, 'lng': -80.1918, 'state': 'FL'}
}

# Top universities
UNIVERSITIES = [
    'Stanford University', 'Harvard University', 'MIT', 'UC Berkeley',
    'Columbia University', 'NYU', 'University of Pennsylvania', 'Yale University',
    'Princeton University', 'Georgetown University', 'Duke University',
    'Northwestern University', 'Cornell University', 'UCLA', 'USC',
    'University of Michigan', 'UT Austin', 'University of Washington',
    'Carnegie Mellon University', 'Brown University', 'Dartmouth College',
    'University of Chicago', 'Johns Hopkins University', 'Vanderbilt University'
]


def generate_bio() -> str:
    """
    Generate realistic user bio with high diversity.

    Target: 18,000+ unique combinations with 500-700 unique words.
    """
    # Expanded vocabulary lists for maximum diversity

    # Action verbs (20 options)
    action_verbs = [
        'Love', 'Enjoy', 'Passionate about', 'Into', 'Obsessed with',
        'Can\'t get enough of', 'Always up for', 'Hooked on', 'Big fan of',
        'Crazy about', 'Addicted to', 'Living for', 'All about', 'Devoted to',
        'Enthusiast for', 'Fascinated by', 'Deep into', 'Really into',
        'Forever loving', 'Committed to'
    ]

    # Activities/hobbies (50 options)
    activities = [
        'hiking', 'traveling', 'cooking', 'photography', 'running', 'yoga',
        'rock climbing', 'surfing', 'skiing', 'snowboarding', 'cycling',
        'swimming', 'dancing', 'painting', 'writing', 'reading', 'gardening',
        'baking', 'wine tasting', 'craft beer hunting', 'coffee roasting',
        'pottery', 'woodworking', 'kayaking', 'camping', 'backpacking',
        'scuba diving', 'sailing', 'tennis', 'golf', 'basketball', 'volleyball',
        'kickboxing', 'pilates', 'meditation', 'filmmaking', 'podcasting',
        'live music', 'concert going', 'museum hopping', 'art galleries',
        'trying new cuisines', 'farmer\'s markets', 'road trips', 'city exploring',
        'beach days', 'mountain adventures', 'trivia nights', 'board games',
        'video gaming', 'stand-up comedy'
    ]

    # Food/drink preferences (30 options)
    food_drinks = [
        'Foodie', 'Coffee enthusiast', 'Wine lover', 'Craft beer fan',
        'Tea connoisseur', 'Whiskey aficionado', 'Brunch devotee', 'Taco lover',
        'Pizza enthusiast', 'Sushi addict', 'Ramen fanatic', 'Burger connoisseur',
        'Vegan chef', 'Home cook', 'BBQ master', 'Cocktail mixer',
        'Smoothie maker', 'Baker at heart', 'Cheese lover', 'Chocolate fiend',
        'Ice cream fanatic', 'Street food hunter', 'Fine dining explorer',
        'Organic food advocate', 'Farm-to-table supporter', 'Dessert first person',
        'Hot sauce collector', 'Coffee snob', 'Wine and dine type', 'Breakfast champion'
    ]

    # Connection goals (25 options)
    connection_goals = [
        'explore the city with', 'try new restaurants with', 'go on adventures with',
        'share good conversations with', 'travel the world with', 'build something with',
        'laugh with', 'grab coffee with', 'catch sunsets with', 'cook meals with',
        'binge Netflix with', 'go to concerts with', 'hit the trails with',
        'discover hidden gems with', 'make memories with', 'share stories with',
        'dream big with', 'get lost with', 'explore museums with', 'try new things with',
        'dance badly with', 'sing karaoke with', 'take photos with', 'plan trips with',
        'debate topics with'
    ]

    # Personality types (30 options)
    personalities = [
        'Adventurer', 'Dreamer', 'Explorer', 'Optimist', 'Realist', 'Idealist',
        'Creative soul', 'Free spirit', 'Old soul', 'Hopeless romantic', 'Pragmatist',
        'Wanderer', 'Thinker', 'Doer', 'Night owl', 'Early bird', 'Introvert',
        'Extrovert', 'Ambivert', 'Minimalist', 'Maximalist', 'Perfectionist',
        'Spontaneous type', 'Planner', 'Risk taker', 'Cautious soul', 'Empath',
        'Analytical mind', 'Artistic soul', 'Renaissance person'
    ]

    # Interests/passions (35 options)
    interests = [
        'Coffee addict', 'Book lover', 'Music junkie', 'Film buff', 'Art enthusiast',
        'History nerd', 'Science geek', 'Tech enthusiast', 'Sports fan', 'Fitness freak',
        'Nature lover', 'Beach bum', 'Mountain person', 'City dweller', 'Dog person',
        'Cat person', 'Plant parent', 'Podcast listener', 'Vinyl collector',
        'Sneaker head', 'Fashion lover', 'Vintage hunter', 'Thrift shopper',
        'DIY enthusiast', 'Sustainability advocate', 'Environmentalist', 'Activist',
        'Volunteer', 'Mentor', 'Lifelong learner', 'Language learner', 'Culture vulture',
        'Astronomy buff', 'Philosophy reader', 'Psychology enthusiast'
    ]

    # Call-to-actions (30 options)
    ctas = [
        'Let\'s grab drinks', 'Coffee first?', 'Tacos and margaritas?', 'Wine bar regular',
        'Brunch this weekend?', 'Let\'s get lost together', 'Up for an adventure?',
        'Show me your city', 'Teach me something new', 'Let\'s make this interesting',
        'Swipe right for good vibes', 'No small talk, let\'s dive deep', 'Let\'s skip the games',
        'Looking for my partner in crime', 'Ready for something real', 'Let\'s grab tacos',
        'Coffee snob seeking same', 'Dog park dates?', 'Museum buddy wanted',
        'Concert companion needed', 'Hiking partner required', 'Foodie friend sought',
        'Travel buddy desired', 'Netflix marathon partner?', 'Gym buddy needed',
        'Book club of two?', 'Debate partner wanted', 'Let\'s create something',
        'Ready to explore?', 'Seeking adventure companion'
    ]

    # Lifestyle phrases (25 options)
    lifestyles = [
        'Into fitness', 'Into wellness', 'Into mindfulness', 'Into personal growth',
        'Focused on balance', 'Living intentionally', 'Chasing dreams', 'Building empire',
        'Finding adventure', 'Seeking growth', 'Embracing change', 'Living fully',
        'Making memories', 'Creating art', 'Pursuing passions', 'Following curiosity',
        'Staying active', 'Keeping grounded', 'Staying positive', 'Living authentically',
        'Being present', 'Choosing joy', 'Spreading kindness', 'Making impact',
        'Living boldly'
    ]

    # Values/qualities (30 options)
    values = [
        'Looking for meaningful connections', 'Here for something real', 'Quality over quantity',
        'No games, just genuine connections', 'Authenticity is key', 'Communication is everything',
        'Honesty above all', 'Loyalty matters', 'Kindness wins', 'Humor required',
        'Intelligence is attractive', 'Ambition is sexy', 'Passion is essential',
        'Adventure is mandatory', 'Growth mindset preferred', 'Emotional intelligence valued',
        'Self-awareness appreciated', 'Confidence is attractive', 'Humility admired',
        'Curiosity encouraged', 'Open-mindedness valued', 'Respect is non-negotiable',
        'Trust is everything', 'Chemistry is crucial', 'Connection over perfection',
        'Substance over surface', 'Depth over breadth', 'Real over perfect',
        'Genuine over filtered', 'Present over past'
    ]

    # Work fields (20 options)
    work_fields = [
        'tech', 'healthcare', 'finance', 'education', 'law', 'consulting',
        'marketing', 'design', 'engineering', 'medicine', 'research', 'nonprofit',
        'government', 'media', 'entertainment', 'hospitality', 'real estate',
        'architecture', 'science', 'arts'
    ]

    # Descriptive adjectives (25 options)
    adjectives = [
        'Love to travel', 'Enjoy the outdoors', 'Foodie at heart', 'Fitness enthusiast',
        'Creative thinker', 'Problem solver', 'Team player', 'Independent spirit',
        'Social butterfly', 'Quiet observer', 'Deep conversationalist', 'Good listener',
        'Storyteller', 'Music lover', 'Art appreciator', 'Book worm', 'Film fanatic',
        'Sports enthusiast', 'Nature lover', 'City explorer', 'Beach person',
        'Mountain type', 'Night person', 'Morning person', 'Weekend warrior'
    ]

    # Location descriptors (15 options)
    location_descriptors = [
        'Born and raised in', 'Currently living in', 'Transplant to', 'Native of',
        'Exploring life in', 'Making home in', 'Recently moved to', 'Loving life in',
        'Building roots in', 'New to', 'Long-time resident of', 'Calling home',
        'Based in', 'Residing in', 'Settled in'
    ]

    # Job attitudes (15 options)
    job_attitudes = [
        'Love my job', 'Love what I do', 'Love my career', 'Passionate about my work',
        'Enjoy my profession', 'Fulfilled by my work', 'Love my craft', 'Driven by my career',
        'Excited about my job', 'Proud of my work', 'Committed to my career',
        'Dedicated to my profession', 'Energized by my work', 'Inspired by my job',
        'Motivated by my career'
    ]

    # Free time activities (25 options)
    freetime_activities = [
        'at the gym', 'trying new restaurants', 'exploring the city', 'at a coffee shop',
        'hiking on weekends', 'at the beach', 'in the mountains', 'at yoga class',
        'reading at parks', 'biking around town', 'at farmers markets', 'cooking at home',
        'walking my dog', 'playing with my cat', 'volunteering', 'at art galleries',
        'catching live music', 'at breweries', 'wine tasting', 'at bookstores',
        'gardening', 'rock climbing', 'surfing', 'skiing', 'traveling'
    ]

    # Partner qualities (30 options)
    partner_qualities = [
        'Sapiosexual', 'Looking for my adventure partner', 'Seeking someone who can keep up with me',
        'Want someone who makes me laugh', 'Need a travel companion', 'Searching for my best friend',
        'Hoping to find my person', 'Looking for genuine connection', 'Seeking my match',
        'Want someone authentic', 'Need someone ambitious', 'Looking for kindred spirit',
        'Seeking intellectual equal', 'Want someone adventurous', 'Need a partner in crime',
        'Looking for my complement', 'Seeking emotional maturity', 'Want mutual growth',
        'Need chemistry and compatibility', 'Looking for deep connection', 'Seeking life partner',
        'Want someone grounded', 'Need someone spontaneous', 'Looking for balance',
        'Seeking passionate soul', 'Want curious mind', 'Need independent spirit',
        'Looking for my co-pilot', 'Seeking someone real', 'Want meaningful bond'
    ]

    # Pet ownership (10 options)
    pets = [
        'Dog lover', 'Cat person', 'Animal lover', 'Dog dad', 'Dog mom',
        'Cat dad', 'Cat mom', 'Pet parent', 'Rescue dog advocate', 'Fur baby parent'
    ]

    # Social preferences (15 options)
    social_prefs = [
        'Netflix and chill?', 'Up for spontaneous trips', 'Always down for brunch',
        'Love a good happy hour', 'Prefer deep conversations', 'Small gatherings over crowds',
        'Intimate dinners preferred', 'Game nights are my jam', 'Concert regular',
        'Festival goer', 'Trivia night champion', 'Karaoke enthusiast', 'Dinner party host',
        'Picnic planner', 'Road trip ready'
    ]

    # Helper function to ensure proper punctuation
    def add_period(text: str) -> str:
        """Add period only if text doesn't already end with punctuation."""
        if text and text[-1] not in '.!?':
            return text + '.'
        return text

    # Templates with MASSIVE variation potential
    templates = [
        # Template 1: Activity-Food-Goal (20 × 50 × 30 × 25 = 750,000 combinations!)
        lambda: f"{random.choice(action_verbs)} {random.choice(activities)}. {random.choice(food_drinks)}. Looking for someone to {random.choice(connection_goals)}.",

        # Template 2: Lifestyle-Pets-Social (30 × 25 × 10 × 15 = 112,500 combinations)
        lambda: add_period(f"{random.choice(personalities)}. {random.choice(lifestyles)}. {random.choice(pets)}. {random.choice(social_prefs)}"),

        # Template 3: Work-Adjective-Value (20 × 25 × 30 = 15,000 combinations)
        lambda: f"Work in {random.choice(work_fields)}. {random.choice(adjectives)}. {random.choice(values)}.",

        # Template 4: Personality-Interest-CTA (30 × 35 × 30 = 31,500 combinations)
        lambda: add_period(f"{random.choice(personalities)}. {random.choice(interests)}. {random.choice(ctas)}"),

        # Template 5: Location-Job-Freetime-Partner (15 × 10 × 10 × 25 × 30 = 1,125,000 combinations)
        lambda: f"{random.choice(location_descriptors)} {random.choice(list(CITIES.keys()))}. {random.choice(job_attitudes)} but also know how to unplug. You can find me {random.choice(freetime_activities)}. {random.choice(partner_qualities)}.",

        # Template 6: NEW - Short and punchy (30 × 50 = 1,500 combinations)
        lambda: f"{random.choice(personalities)} seeking {random.choice(activities)} partner.",

        # Template 7: NEW - Interest stack (35 × 35 × 30 = 36,750 combinations)
        lambda: add_period(f"{random.choice(interests)}. {random.choice(interests)}. {random.choice(ctas)}"),

        # Template 8: NEW - Value-driven (30 × 30 × 25 = 22,500 combinations)
        lambda: f"{random.choice(values)}. {random.choice(adjectives)}. Looking for someone to {random.choice(connection_goals)}.",

        # Template 9: NEW - Activity focus (20 × 50 × 50 = 50,000 combinations)
        lambda: add_period(f"{random.choice(action_verbs)} {random.choice(activities)} and {random.choice(activities)}. {random.choice(ctas)}"),

        # Template 10: NEW - Lifestyle description (25 × 30 × 30 = 22,500 combinations)
        lambda: f"{random.choice(lifestyles)}. {random.choice(food_drinks)}. {random.choice(partner_qualities)}."
    ]

    # TOTAL theoretical combinations: ~2,167,250 possible unique bios!
    # For 20,000 users, we expect ~99.1% uniqueness with random sampling

    return random.choice(templates)()


def get_job_title() -> Tuple[str, str]:
    """Return (industry, job_title) based on weighted distribution."""
    industry = random.choices(
        list(INDUSTRIES.keys()),
        weights=[v['weight'] for v in INDUSTRIES.values()]
    )[0]
    job_title = random.choice(INDUSTRIES[industry]['titles'])
    return industry, job_title


def generate_users(num_users: int) -> pd.DataFrame:
    """Generate user data."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Generating {num_users} users...")

    users = []

    for i in range(num_users):
        if (i + 1) % 5000 == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Generated {i + 1} users...")

        user_id = str(uuid.uuid4())

        # Basic demographics
        gender = random.choice(['male', 'female'])
        age = random.randint(25, 45)
        birthday = (datetime.now() - timedelta(days=age * 365 + random.randint(0, 365))).strftime('%m/%d/%Y')

        # Account creation (staggered over 1 year)
        days_since_join = random.randint(0, 365)
        created_at = APP_START_DATE + timedelta(days=days_since_join)

        # Premium status
        is_premium = random.random() < PREMIUM_RATE
        is_super = random.random() < SUPER_RATE

        # Hearts (daily swipes)
        hearts = random.randint(50, 200) if is_premium else random.randint(0, 50)

        # Swipes left (daily limit usage)
        swipes_left = random.randint(0, 10)

        # Location
        city = random.choice(list(CITIES.keys()))
        city_data = CITIES[city]
        # Add some noise to coordinates for variety
        lat = city_data['lat'] + random.uniform(-0.1, 0.1)
        lng = city_data['lng'] + random.uniform(-0.1, 0.1)
        geohash = pgh.encode(lat, lng, precision=7)

        # Professional info
        industry, job_title = get_job_title()
        school = random.choice(UNIVERSITIES)

        # Activity - determine if user is active
        # Users who joined recently are more likely to be active
        if days_since_join < 30:
            is_active = random.random() < 0.9  # 90% active
        elif days_since_join < 90:
            is_active = random.random() < 0.7  # 70% active
        elif days_since_join < 180:
            is_active = random.random() < 0.5  # 50% active
        else:
            is_active = random.random() < 0.3  # 30% active

        # Last login
        if is_active:
            days_since_login = random.randint(0, min(7, days_since_join))
        else:
            # Ensure inactive users have login at least 30 days ago, but not before they joined
            max_days_ago = 365 - days_since_join
            if max_days_ago >= 30:
                days_since_login = random.randint(30, max_days_ago)
            else:
                days_since_login = max_days_ago

        # Last login is somewhere between join date and (now - days_since_login)
        max_login_offset = max(0, 365 - days_since_join - days_since_login)
        last_login = created_at + timedelta(days=random.randint(0, max_login_offset) if max_login_offset > 0 else 0)

        # Matching preferences
        preferred_gender = random.choice(['male', 'female', 'both'])
        age_range_width = random.randint(5, 15)
        min_age = max(25, age - age_range_width // 2)
        max_age = min(45, age + age_range_width // 2)
        use_location = random.random() < 0.8  # 80% use location filter

        user = {
            'user_id': user_id,
            'name': fake.first_name(),
            'email': fake.email(),
            'email_verified': random.random() < 0.95,  # 95% verified
            'bio': generate_bio(),
            'birthday': birthday,
            'age': age,
            'gender': gender,
            'country': 1,  # US
            'created_at': created_at.isoformat(),
            'last_login': last_login.isoformat(),
            'is_premium': is_premium,
            'is_super': is_super,
            'hearts': hearts,
            'swipes_left': swipes_left,
            'job_title': job_title,
            'industry': industry,
            'school': school,
            'city': city,
            'state': city_data['state'],
            'latitude': lat,
            'longitude': lng,
            'geohash': geohash,
            'matching_pref_age_min': min_age,
            'matching_pref_age_max': max_age,
            'matching_pref_gender': preferred_gender,
            'matching_pref_use_location': use_location,
            'is_active': is_active,
            'days_since_join': days_since_join
        }

        users.append(user)

    df = pd.DataFrame(users)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] User generation complete!")
    return df


def calculate_compatibility(user1: Dict, user2: Dict) -> float:
    """Calculate compatibility score between two users (0-1)."""
    score = 0.0

    # Gender preference match (required)
    if user1['matching_pref_gender'] == 'both' or user1['matching_pref_gender'] == user2['gender']:
        if user2['matching_pref_gender'] == 'both' or user2['matching_pref_gender'] == user1['gender']:
            score = 0.3  # Base score for gender match
        else:
            return 0.0  # No compatibility if gender doesn't match
    else:
        return 0.0

    # Age compatibility
    if user1['matching_pref_age_min'] <= user2['age'] <= user1['matching_pref_age_max']:
        score += 0.2

    if user2['matching_pref_age_min'] <= user1['age'] <= user2['matching_pref_age_max']:
        score += 0.2

    # Same city (strong signal)
    if user1['city'] == user2['city']:
        score += 0.2

    # Similar education level (same university or tier)
    if user1['school'] == user2['school']:
        score += 0.1

    return min(score, 1.0)


def generate_ultra_concentrated_likes(users_df, target_likes=2000000, seed=42):
    """
    Generate likes with RECIPROCAL PATTERNS and TEMPORAL CONSISTENCY.

    KEY FEATURES:
    - TWO-PASS GENERATION: 70% initial preference-driven likes, 30% reciprocal likes
    - RECIPROCAL PROBABILITY: 1-15% chance to like back (based on compatibility)
    - 3-5 MEGA-CLUSTERS (4000-6000 users each) for strong CF signal
    - EXPONENTIAL preference weighting - amplifies score differences
    - Top 10% "popular" users receive 40% of ALL likes (power law)
    - 60% of likes concentrated in TOP 10% of scored candidates (extreme focus)
    - 25% in TOP 30% (moderate focus)
    - 10% in TOP 50% (exploration)
    - 5% random (diversity)
    - Minimum 30 likes/user (ensures robust test set after 80/20 split)
    - CLUSTER-ONLY sampling for maximum concentration
    - **TEMPORAL CONSISTENCY**: User preferences stable over time (30-50% train/test overlap)

    Expected Results:
    - User overlap: 20-35% (vs previous 0.5%)
    - CF Precision@10: 5-15% (vs previous 0.3%)
    - Match rate: 0.8-1.5% (vs previous 0.3%)
    - Reciprocal rate: 25-35% of all likes
    - Train/Test overlap: 30-50% per user (vs previous 0%)
    """
    np.random.seed(seed)
    random.seed(seed)

    print("\n" + "="*80)
    print("GENERATING LIKES WITH RECIPROCAL PATTERNS (TWO-SIDED MATCHING)")
    print("="*80)

    # Layer 1: Create 3-5 MEGA-CLUSTERS with extreme concentration
    print("\n[1/6] Creating mega-clusters...")
    users_df = users_df.copy()

    # Strategy: Group cities into 3-5 mega-clusters
    # 10 cities → 3-5 groups based on city codes
    city_codes = users_df['city'].astype('category').cat.codes
    users_df['mega_cluster'] = city_codes // 2  # 10 cities → 5 clusters

    num_clusters = users_df['mega_cluster'].nunique()
    cluster_sizes = users_df.groupby('mega_cluster').size()
    print(f"✓ Created {num_clusters} MEGA-CLUSTERS")
    print(f"✓ Average cluster size: {cluster_sizes.mean():.0f} users")
    print(f"✓ Range: {cluster_sizes.min()}-{cluster_sizes.max()} users per cluster")
    
    # Layer 2: Identify "popular" users (top 10% will get 40% of likes)
    print("\n[2/6] Identifying popular users for power law distribution...")

    # Select 10% of users as "popular" (balanced by gender and cluster)
    popular_users = set()
    n_popular_per_cluster_gender = max(1, int(0.10 * len(users_df) / (num_clusters * 2)))

    for cluster_id in range(num_clusters):
        for gender in ['male', 'female']:
            cluster_gender_users = users_df[
                (users_df['mega_cluster'] == cluster_id) &
                (users_df['gender'] == gender)
            ]
            # Randomly select popular users
            n_popular = min(n_popular_per_cluster_gender, len(cluster_gender_users))
            popular_sample = cluster_gender_users.sample(n=n_popular, random_state=seed+cluster_id)
            popular_users.update(popular_sample['user_id'].tolist())

    users_df['is_popular'] = users_df['user_id'].isin(popular_users)
    print(f"✓ Selected {len(popular_users)} popular users ({len(popular_users)/len(users_df)*100:.1f}%)")
    print(f"✓ These users will receive ~40% of all likes")

    # Layer 3: Compute NLP similarity for attraction
    print("\n[3/6] Computing profile similarity (NLP-based attraction)...")
    print("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    profile_texts = []
    for _, row in users_df.iterrows():
        text = f"{row['bio']} {row['job_title']} {row['school']} in {row['city']}"
        profile_texts.append(text)

    print("Generating embeddings (this may take 2-3 minutes)...")
    embeddings = model.encode(profile_texts, batch_size=128, show_progress_bar=True)
    print(f"✓ Embeddings shape: {embeddings.shape}")

    # Layer 4: PASS 1 - Generate initial preference-driven likes with TEMPORAL CONSISTENCY
    print("\n[4/6] PASS 1: Generating initial preference-driven likes with temporal consistency...")
    likes_data = []
    incoming_likes = {}  # Track who liked whom: {target_user_id: [list of liker_user_ids]}

    # Store user preference scores for temporal consistency
    # Key: user_id, Value: dict of {candidate_id: preference_score}
    user_preference_scores = {}

    # Calculate activity per user: 70 avg likes/user with minimum 20
    # (This is 70% of final target, remaining 30% comes from reciprocal pass)
    activity_distribution = np.random.lognormal(4.1, 0.6, len(users_df))

    for idx in tqdm(range(len(users_df)), desc="Pass 1 - Initial likes with temporal split"):
        row = users_df.iloc[idx]
        user_id = row['user_id']
        user_cluster = row['mega_cluster']
        user_gender = row['gender']
        user_age = row['age']
        user_embedding = embeddings[idx]

        # Minimum 20 likes for Pass 1 (will add more in Pass 2)
        num_user_likes = int(activity_distribution[idx])
        num_user_likes = max(20, min(num_user_likes, 140))  # Min 20, max 140

        # CRITICAL: Get candidates from SAME CLUSTER ONLY (maximum concentration!)
        if user_gender == 'male':
            gender_mask = users_df['gender'] == 'female'
        else:
            gender_mask = users_df['gender'] == 'male'

        # SAME CLUSTER ONLY - this is the key to high overlap!
        cluster_mask = users_df['mega_cluster'] == user_cluster

        # Relaxed age filter for more candidates
        age_mask = (
            (users_df['age'] >= user_age - 20) &
            (users_df['age'] <= user_age + 20)
        )

        candidates_mask = gender_mask & cluster_mask & age_mask
        candidates = users_df[candidates_mask].copy()
        
        if len(candidates) == 0:
            continue

        # Compute base preference scores
        candidate_indices = candidates.index.tolist()
        candidate_embeddings = embeddings[candidate_indices]

        # NLP similarity (0-1 range)
        similarity_scores = cosine_similarity([user_embedding], candidate_embeddings)[0]

        # Popular user bonus (HUGE impact!)
        popularity_bonus = candidates['is_popular'].astype(float).values * 1.5  # 1.5 bonus for popular users!

        # Combined base scores (both are numpy arrays)
        base_scores = similarity_scores + popularity_bonus

        # EXPONENTIAL weighting (KEY IMPROVEMENT!)
        # This amplifies score differences dramatically
        preference_scores = np.exp(base_scores * 1.5)  # Exponential with scaling factor

        # Normalize to probabilities
        score_sum = preference_scores.sum()
        if score_sum == 0:
            score_sum = 1.0
        probs = preference_scores / score_sum

        # ULTRA-CONCENTRATED SAMPLING STRATEGY
        # 60% from TOP 10% of candidates (extreme concentration!)
        # 25% from TOP 30% (moderate focus)
        # 10% from TOP 50% (exploration)
        # 5% random (diversity)

        n_candidates = len(candidates)
        top_10_pct = max(1, int(n_candidates * 0.10))
        top_30_pct = max(top_10_pct + 1, int(n_candidates * 0.30))
        top_50_pct = max(top_30_pct + 1, int(n_candidates * 0.50))

        # Get top indices by score
        sorted_indices = np.argsort(preference_scores)[::-1]

        # Calculate number of likes from each tier
        n_from_top10 = int(num_user_likes * 0.60)
        n_from_top30 = int(num_user_likes * 0.25)
        n_from_top50 = int(num_user_likes * 0.10)
        n_random = num_user_likes - n_from_top10 - n_from_top30 - n_from_top50

        liked_indices = []

        # Sample from TOP 10%
        top10_candidates = sorted_indices[:top_10_pct]
        if len(top10_candidates) > 0:
            n_sample = min(n_from_top10, len(top10_candidates))
            sampled = np.random.choice(top10_candidates, size=n_sample, replace=False)
            liked_indices.extend(sampled)

        # Sample from TOP 30% (excluding already sampled)
        top30_candidates = sorted_indices[top_10_pct:top_30_pct]
        if len(top30_candidates) > 0:
            n_sample = min(n_from_top30, len(top30_candidates))
            sampled = np.random.choice(top30_candidates, size=n_sample, replace=False)
            liked_indices.extend(sampled)

        # Sample from TOP 50% (excluding already sampled)
        top50_candidates = sorted_indices[top_30_pct:top_50_pct]
        if len(top50_candidates) > 0:
            n_sample = min(n_from_top50, len(top50_candidates))
            sampled = np.random.choice(top50_candidates, size=n_sample, replace=False)
            liked_indices.extend(sampled)

        # Random sampling from rest
        rest_candidates = sorted_indices[top_50_pct:]
        if len(rest_candidates) > 0 and n_random > 0:
            n_sample = min(n_random, len(rest_candidates))
            sampled = np.random.choice(rest_candidates, size=n_sample, replace=False)
            liked_indices.extend(sampled)

        # Convert to numpy array
        liked_indices = np.array(liked_indices)

        # TEMPORAL CONSISTENCY: Store preference scores for this user
        # This allows us to generate consistent likes across train/test periods
        candidate_ids = [candidates.iloc[i]['user_id'] for i in liked_indices]
        user_pref_scores = {
            candidate_ids[i]: preference_scores[liked_indices[i]]
            for i in range(len(liked_indices))
        }
        user_preference_scores[user_id] = user_pref_scores

        # Create like records with TEMPORAL SPLIT
        user_created_at = datetime.fromisoformat(row['created_at'])

        # User's active period (from join to now)
        days_active = min(365, (datetime.now() - user_created_at).days)
        if days_active == 0:
            days_active = 1

        # Split timeline: 80% training period, 20% test period
        train_cutoff_day = int(days_active * 0.80)

        # Split likes into train/test with temporal consistency
        # 80% of likes go to training period, 20% to test period
        n_train_likes = int(len(liked_indices) * 0.80)
        n_test_likes = len(liked_indices) - n_train_likes

        # CRITICAL: Use the SAME candidates for both periods (ensures overlap!)
        # Take top candidates by preference score to ensure they appear in both
        sorted_candidate_indices = np.argsort(preference_scores[liked_indices])[::-1]

        # Training period likes (days 0 to train_cutoff_day)
        for i in range(n_train_likes):
            cand_local_idx = sorted_candidate_indices[i]
            cand_idx = liked_indices[cand_local_idx]
            candidate = candidates.iloc[cand_local_idx]
            target_id = candidate['user_id']

            base_score = base_scores[cand_idx]
            action = 'superlike' if base_score > 1.8 else 'like'

            # Timestamp in training period
            like_day = np.random.randint(0, max(1, train_cutoff_day))
            like_time = user_created_at + timedelta(days=like_day)

            likes_data.append({
                'like_id': str(uuid.uuid4()),
                'user_id': user_id,
                'liked_user_id': target_id,
                'timestamp': like_time.isoformat(),
                'action': action
            })

            if target_id not in incoming_likes:
                incoming_likes[target_id] = []
            incoming_likes[target_id].append((user_id, like_time.isoformat(), action))

        # Test period likes (days train_cutoff_day+1 to days_active)
        # OVERLAP STRATEGY: 40% exact same candidates, 60% new (but similar) candidates
        n_overlap = int(n_test_likes * 0.40)  # 40% overlap for 30-50% train/test consistency
        n_new = n_test_likes - n_overlap

        # Overlap likes: Use candidates from training period (top preference scores)
        for i in range(n_overlap):
            if i >= n_train_likes:  # Safety check
                break
            cand_local_idx = sorted_candidate_indices[i]  # Reuse same top candidates
            cand_idx = liked_indices[cand_local_idx]
            candidate = candidates.iloc[cand_local_idx]
            target_id = candidate['user_id']

            base_score = base_scores[cand_idx]
            action = 'superlike' if base_score > 1.8 else 'like'

            # Timestamp in test period
            test_period_days = days_active - train_cutoff_day
            if test_period_days > 0:
                like_day = train_cutoff_day + np.random.randint(0, test_period_days)
            else:
                like_day = train_cutoff_day
            like_time = user_created_at + timedelta(days=like_day)

            likes_data.append({
                'like_id': str(uuid.uuid4()),
                'user_id': user_id,
                'liked_user_id': target_id,
                'timestamp': like_time.isoformat(),
                'action': action
            })

            if target_id not in incoming_likes:
                incoming_likes[target_id] = []
            incoming_likes[target_id].append((user_id, like_time.isoformat(), action))

        # New likes in test period: Similar candidates (next best by preference score)
        for i in range(n_new):
            offset = n_train_likes + i
            if offset >= len(sorted_candidate_indices):
                break
            cand_local_idx = sorted_candidate_indices[offset]
            cand_idx = liked_indices[cand_local_idx]
            candidate = candidates.iloc[cand_local_idx]
            target_id = candidate['user_id']

            base_score = base_scores[cand_idx]
            action = 'superlike' if base_score > 1.8 else 'like'

            # Timestamp in test period
            test_period_days = days_active - train_cutoff_day
            if test_period_days > 0:
                like_day = train_cutoff_day + np.random.randint(0, test_period_days)
            else:
                like_day = train_cutoff_day
            like_time = user_created_at + timedelta(days=like_day)

            likes_data.append({
                'like_id': str(uuid.uuid4()),
                'user_id': user_id,
                'liked_user_id': target_id,
                'timestamp': like_time.isoformat(),
                'action': action
            })

            if target_id not in incoming_likes:
                incoming_likes[target_id] = []
            incoming_likes[target_id].append((user_id, like_time.isoformat(), action))

    print(f"\nPass 1 Complete: Generated {len(likes_data):,} initial likes")
    print(f"Average likes per user: {len(likes_data) / len(users_df):.1f}")
    print(f"Users with incoming likes: {len(incoming_likes):,}")
    print(f"Avg incoming likes per user with likes: {np.mean([len(v) for v in incoming_likes.values()]):.1f}")

    # Layer 5: PASS 2 - Generate reciprocal likes (high probability of liking back)
    print("\n[5/6] PASS 2: Generating reciprocal likes (liking back)...")

    reciprocal_likes_count = 0

    for idx in tqdm(range(len(users_df)), desc="Pass 2 - Reciprocal likes"):
        row = users_df.iloc[idx]
        user_id = row['user_id']
        user_cluster = row['mega_cluster']
        user_age = row['age']
        user_city = row['city']
        user_embedding = embeddings[idx]

        # Check if this user has received any likes
        if user_id not in incoming_likes or len(incoming_likes[user_id]) == 0:
            continue

        # Get list of users who liked this user
        potential_reciprocals = incoming_likes[user_id]

        # Check which ones this user HASN'T already liked back
        already_liked = set([like['liked_user_id'] for like in likes_data if like['user_id'] == user_id])
        potential_reciprocals = [(liker_id, ts, action) for liker_id, ts, action in potential_reciprocals
                                 if liker_id not in already_liked]

        if len(potential_reciprocals) == 0:
            continue

        # For each incoming like, compute reciprocal probability
        for liker_user_id, original_timestamp, original_action in potential_reciprocals:
            # Get liker's info
            liker_idx = users_df[users_df['user_id'] == liker_user_id].index
            if len(liker_idx) == 0:
                continue
            liker_idx = liker_idx[0]
            liker_row = users_df.iloc[liker_idx]
            liker_embedding = embeddings[liker_idx]

            # Base reciprocal probability (VERY low to achieve realistic 0.8-1.5% match rates)
            # In real dating apps, most likes don't get reciprocated
            reciprocal_prob = 0.01  # Only 1% base probability

            # Boost based on compatibility factors (but still very conservative)
            # Same cluster bonus (strongest signal)
            if liker_row['mega_cluster'] == user_cluster:
                reciprocal_prob += 0.02

            # NLP similarity bonus
            similarity = cosine_similarity([user_embedding], [liker_embedding])[0][0]
            if similarity > 0.80:
                reciprocal_prob += 0.05
            elif similarity > 0.75:
                reciprocal_prob += 0.03
            elif similarity > 0.70:
                reciprocal_prob += 0.01

            # Age compatibility bonus
            age_diff = abs(liker_row['age'] - user_age)
            if age_diff < 3:
                reciprocal_prob += 0.02
            elif age_diff < 5:
                reciprocal_prob += 0.01

            # Location bonus
            if liker_row['city'] == user_city:
                reciprocal_prob += 0.01

            # Popular user bonus (people are more likely to like back popular users)
            if liker_row['is_popular']:
                reciprocal_prob += 0.03

            # Cap at 15% (realistic maximum for dating apps)
            reciprocal_prob = min(reciprocal_prob, 0.15)

            # Decide whether to like back
            if np.random.random() < reciprocal_prob:
                # Determine action type (higher superlike rate for reciprocal likes)
                # If original liker superliked, 30% chance to superlike back
                if original_action == 'superlike':
                    action_type = 'superlike' if np.random.random() < 0.30 else 'like'
                else:
                    action_type = 'superlike' if np.random.random() < 0.08 else 'like'

                # Generate timestamp (after the original like)
                original_dt = datetime.fromisoformat(original_timestamp)
                # Typically respond within 1-2 weeks (exponential distribution)
                days_after = int(np.random.exponential(7))
                days_after = min(days_after, 30)  # Cap at 30 days
                timestamp = original_dt + timedelta(days=days_after)

                likes_data.append({
                    'like_id': str(uuid.uuid4()),
                    'user_id': user_id,
                    'liked_user_id': liker_user_id,
                    'timestamp': timestamp.isoformat(),
                    'action': action_type
                })

                reciprocal_likes_count += 1

    print(f"\nPass 2 Complete: Generated {reciprocal_likes_count:,} reciprocal likes")
    print(f"Reciprocal rate: {reciprocal_likes_count / len(likes_data) * 100:.1f}% of initial likes")

    # Convert all likes to DataFrame
    likes_df = pd.DataFrame(likes_data)

    print(f"\n{'='*80}")
    print(f"TOTAL LIKES GENERATED")
    print(f"{'='*80}")
    print(f"Total likes: {len(likes_df):,}")
    print(f"Average likes per user: {len(likes_df) / len(users_df):.1f}")
    print(f"Superlike rate: {(likes_df['action'] == 'superlike').sum() / len(likes_df) * 100:.1f}%")

    # Calculate and display overlap statistics
    print(f"\nCollaborative Filtering Signal Analysis:")
    sample_users = np.random.choice(users_df['user_id'].values, size=min(100, len(users_df)), replace=False)
    user_likes_dict = likes_df[likes_df['user_id'].isin(sample_users)].groupby('user_id')['liked_user_id'].apply(set).to_dict()

    if len(user_likes_dict) >= 2:
        overlap_scores = []
        user_list = list(user_likes_dict.keys())
        for i in range(min(20, len(user_list))):
            for j in range(i+1, min(20, len(user_list))):
                likes_i = user_likes_dict[user_list[i]]
                likes_j = user_likes_dict[user_list[j]]
                if len(likes_i) > 0 and len(likes_j) > 0:
                    overlap = len(likes_i & likes_j) / min(len(likes_i), len(likes_j))
                    overlap_scores.append(overlap)

        if overlap_scores:
            avg_overlap = np.mean(overlap_scores)
            print(f"Average user overlap (Jaccard): {avg_overlap:.1%}")
            if avg_overlap >= 0.20:
                print(f"  EXCELLENT! Strong CF signal (target: >=20%)")
            elif avg_overlap >= 0.15:
                print(f"  GOOD! Moderate CF signal (target: >=20%)")
            elif avg_overlap >= 0.10:
                print(f"  FAIR. Weak CF signal (target: >=20%)")
            else:
                print(f"  POOR. Very weak CF signal (target: >=20%)")

    # Layer 6: Analyze reciprocal patterns
    print(f"\n{'='*80}")
    print("RECIPROCAL PATTERN ANALYSIS")
    print(f"{'='*80}")

    # Calculate reciprocal pairs (both users liked each other)
    like_pairs = {}
    for _, row in likes_df.iterrows():
        pair = tuple(sorted([row['user_id'], row['liked_user_id']]))
        if pair not in like_pairs:
            like_pairs[pair] = []
        like_pairs[pair].append(row)

    reciprocal_pairs = sum(1 for pair_likes in like_pairs.values() if len(pair_likes) >= 2)
    total_pairs = len(like_pairs)

    print(f"✓ Total unique user pairs with at least one like: {total_pairs:,}")
    print(f"✓ Reciprocal pairs (both liked each other): {reciprocal_pairs:,}")
    print(f"✓ Reciprocal pair rate: {reciprocal_pairs / total_pairs * 100:.1f}%")
    print(f"✓ Expected match rate from reciprocals: {reciprocal_pairs * 2 / len(likes_df) * 100:.2f}%")

    # Layer 6.5: Validate temporal consistency
    print(f"\n{'='*80}")
    print("TEMPORAL CONSISTENCY VALIDATION")
    print(f"{'='*80}")

    # Split likes into train/test by timestamp
    likes_df_temp = pd.DataFrame(likes_data)
    likes_df_temp['timestamp_dt'] = pd.to_datetime(likes_df_temp['timestamp'])

    # For each user, compute train/test split
    overlap_scores = []
    for user_id in likes_df_temp['user_id'].unique()[:100]:  # Sample 100 users
        user_likes = likes_df_temp[likes_df_temp['user_id'] == user_id]
        if len(user_likes) < 10:  # Need sufficient data
            continue

        # Temporal split at 80%
        user_likes_sorted = user_likes.sort_values('timestamp_dt')
        split_idx = int(len(user_likes_sorted) * 0.80)

        train_likes_set = set(user_likes_sorted.iloc[:split_idx]['liked_user_id'])
        test_likes_set = set(user_likes_sorted.iloc[split_idx:]['liked_user_id'])

        if len(test_likes_set) > 0:
            overlap = len(train_likes_set & test_likes_set) / len(test_likes_set)
            overlap_scores.append(overlap)

    if overlap_scores:
        avg_overlap = np.mean(overlap_scores)
        print(f"Average train/test overlap: {avg_overlap:.1%}")
        print(f"Min overlap: {np.min(overlap_scores):.1%}")
        print(f"Max overlap: {np.max(overlap_scores):.1%}")
        print(f"Median overlap: {np.median(overlap_scores):.1%}")

        if avg_overlap >= 0.30:
            print(f"  EXCELLENT! Strong temporal consistency (target: >=30%)")
        elif avg_overlap >= 0.20:
            print(f"  GOOD! Moderate temporal consistency (target: >=30%)")
        elif avg_overlap >= 0.10:
            print(f"  FAIR. Weak temporal consistency (target: >=30%)")
        else:
            print(f"  POOR. Very weak temporal consistency (target: >=30%)")

    # Layer 7: Generate matches from mutual likes
    print(f"\n{'='*80}")
    print("[6/6] Generating matches from mutual likes...")
    print(f"{'='*80}\n")

    # Build mutual like pairs
    like_pairs = defaultdict(set)
    like_timestamps = {}

    for _, like in likes_df.iterrows():
        user_id = like['user_id']
        liked_id = like['liked_user_id']
        timestamp = like['timestamp']

        like_pairs[user_id].add(liked_id)
        like_timestamps[(user_id, liked_id)] = timestamp

    # Find mutual likes
    matches = []
    processed_pairs = set()

    for user_id, liked_users in like_pairs.items():
        for liked_id in liked_users:
            # Check if it's a mutual like
            if liked_id in like_pairs and user_id in like_pairs[liked_id]:
                # Create a canonical pair (sorted) to avoid duplicates
                pair = tuple(sorted([user_id, liked_id]))

                if pair in processed_pairs:
                    continue

                processed_pairs.add(pair)

                # Match timestamp is the later of the two likes
                ts1 = datetime.fromisoformat(like_timestamps[(user_id, liked_id)])
                ts2 = datetime.fromisoformat(like_timestamps[(liked_id, user_id)])
                matched_at = max(ts1, ts2)

                # 70% of matches have conversations
                has_conversation = random.random() < 0.70

                if has_conversation:
                    # Message count follows power law: most have 1-10, few have 30-50
                    rand = random.random()
                    if rand < 0.70:  # 70% have 1-10 messages
                        message_count = random.randint(1, 10)
                    elif rand < 0.90:  # 20% have 11-20 messages
                        message_count = random.randint(11, 20)
                    else:  # 10% have 21-50 messages
                        message_count = random.randint(21, 50)

                    # Last message within days/weeks after match
                    days_after = random.randint(0, 30)
                    last_message_at = matched_at + timedelta(days=days_after)
                else:
                    message_count = 0
                    last_message_at = None

                matches.append({
                    'match_id': str(uuid.uuid4()),
                    'user1_id': pair[0],
                    'user2_id': pair[1],
                    'matched_at': matched_at.isoformat(),
                    'conversation_started': has_conversation,
                    'message_count': message_count,
                    'last_message_at': last_message_at.isoformat() if last_message_at else None
                })

    matches_df = pd.DataFrame(matches)

    print(f"✓ Generated {len(matches_df):,} matches")
    if len(likes_df) > 0:
        match_rate = (len(matches_df) * 2) / len(likes_df) * 100
        print(f"✓ Match rate: {match_rate:.2f}%")

    return likes_df, matches_df


def generate_enhanced_likes(users_df, target_likes=2400000, seed=42):
    """
    DEPRECATED: Use generate_ultra_concentrated_likes() instead.

    This function is kept for backwards compatibility but produces
    data with only 0.5% user overlap, causing poor CF performance.

    See generate_ultra_concentrated_likes() for the improved version
    that achieves 20-35% overlap.
    """
    print("\nWARNING: Using deprecated generate_enhanced_likes()")
    print("This produces weak CF signals (0.5% overlap)")
    print("Use generate_ultra_concentrated_likes() instead for 20-35% overlap\n")
    return generate_ultra_concentrated_likes(users_df, target_likes, seed)


def generate_matches(users_df: pd.DataFrame, likes_df: pd.DataFrame) -> pd.DataFrame:
    """Generate matches from mutual likes."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Generating matches from mutual likes...")

    # Build mutual like pairs
    like_pairs = defaultdict(set)
    like_timestamps = {}

    for _, like in likes_df.iterrows():
        user_id = like['user_id']
        liked_id = like['liked_user_id']
        timestamp = like['timestamp']

        like_pairs[user_id].add(liked_id)
        like_timestamps[(user_id, liked_id)] = timestamp

    # Find mutual likes
    matches = []
    processed_pairs = set()

    for user_id, liked_users in like_pairs.items():
        for liked_id in liked_users:
            # Check if it's a mutual like
            if liked_id in like_pairs and user_id in like_pairs[liked_id]:
                # Create a canonical pair (sorted) to avoid duplicates
                pair = tuple(sorted([user_id, liked_id]))

                if pair in processed_pairs:
                    continue

                processed_pairs.add(pair)

                # Match timestamp is the later of the two likes
                ts1 = datetime.fromisoformat(like_timestamps[(user_id, liked_id)])
                ts2 = datetime.fromisoformat(like_timestamps[(liked_id, user_id)])
                matched_at = max(ts1, ts2)

                # 70% of matches have conversations
                has_conversation = random.random() < 0.70

                if has_conversation:
                    # Message count follows power law: most have 1-10, few have 30-50
                    rand = random.random()
                    if rand < 0.70:  # 70% have 1-10 messages
                        message_count = random.randint(1, 10)
                    elif rand < 0.90:  # 20% have 11-20 messages
                        message_count = random.randint(11, 20)
                    else:  # 10% have 21-50 messages
                        message_count = random.randint(21, 50)

                    # Last message within days/weeks after match
                    days_after = random.randint(0, 30)
                    last_message_at = matched_at + timedelta(days=days_after)
                else:
                    message_count = 0
                    last_message_at = None

                matches.append({
                    'match_id': str(uuid.uuid4()),
                    'user1_id': pair[0],
                    'user2_id': pair[1],
                    'matched_at': matched_at.isoformat(),
                    'conversation_started': has_conversation,
                    'message_count': message_count,
                    'last_message_at': last_message_at.isoformat() if last_message_at else None
                })

    matches_df = pd.DataFrame(matches)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Match generation complete! Total: {len(matches_df)}")

    # Calculate match rate
    if len(likes_df) > 0:
        match_rate = (len(matches_df) * 2) / len(likes_df) * 100  # *2 because each match = 2 likes
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Match rate: {match_rate:.2f}%")

    return matches_df


def validate_and_summarize(users_df: pd.DataFrame, likes_df: pd.DataFrame, matches_df: pd.DataFrame):
    """Validate data and print summary statistics."""
    print(f"\n{'='*60}")
    print(f"DATA VALIDATION AND SUMMARY")
    print(f"{'='*60}\n")

    # Basic counts
    print(f"DATASET SIZES:")
    print(f"  Total Users: {len(users_df):,}")
    print(f"  Total Likes: {len(likes_df):,}")
    print(f"  Total Matches: {len(matches_df):,}")
    print(f"  Match Rate: {(len(matches_df) * 2 / len(likes_df) * 100):.2f}%\n")

    # User demographics
    print(f"USER DEMOGRAPHICS:")
    print(f"  Gender Distribution:")
    for gender, count in users_df['gender'].value_counts().items():
        print(f"    {gender}: {count:,} ({count/len(users_df)*100:.1f}%)")

    print(f"\n  Premium Users: {users_df['is_premium'].sum():,} ({users_df['is_premium'].sum()/len(users_df)*100:.1f}%)")
    print(f"  Super Users: {users_df['is_super'].sum():,} ({users_df['is_super'].sum()/len(users_df)*100:.1f}%)")
    print(f"  Active Users: {users_df['is_active'].sum():,} ({users_df['is_active'].sum()/len(users_df)*100:.1f}%)")

    print(f"\n  Age Range: {users_df['age'].min()}-{users_df['age'].max()} (avg: {users_df['age'].mean():.1f})")

    # Industry distribution
    print(f"\n  Industry Distribution:")
    for industry, count in users_df['industry'].value_counts().head(10).items():
        print(f"    {industry}: {count:,} ({count/len(users_df)*100:.1f}%)")

    # City distribution
    print(f"\n  Top Cities:")
    for city, count in users_df['city'].value_counts().head(10).items():
        print(f"    {city}: {count:,} ({count/len(users_df)*100:.1f}%)")

    # Like patterns
    print(f"\nLIKE PATTERNS:")
    likes_per_user = likes_df.groupby('user_id').size()
    print(f"  Avg Likes per User: {likes_per_user.mean():.1f}")
    print(f"  Median Likes per User: {likes_per_user.median():.0f}")
    print(f"  Max Likes by User: {likes_per_user.max()}")
    print(f"  Min Likes by User: {likes_per_user.min()}")

    # Power law check
    print(f"\n  Power Law Distribution:")
    q90 = likes_per_user.quantile(0.90)
    q75 = likes_per_user.quantile(0.75)
    q50 = likes_per_user.quantile(0.50)
    print(f"    Top 10% users: >{q90:.0f} likes")
    print(f"    Top 25% users: >{q75:.0f} likes")
    print(f"    Median: {q50:.0f} likes")

    # Action types
    print(f"\n  Action Distribution:")
    for action, count in likes_df['action'].value_counts().items():
        print(f"    {action}: {count:,} ({count/len(likes_df)*100:.1f}%)")

    # Match patterns
    print(f"\nMATCH PATTERNS:")
    print(f"  Matches with Conversation: {matches_df['conversation_started'].sum():,} ({matches_df['conversation_started'].sum()/len(matches_df)*100:.1f}%)")

    conv_matches = matches_df[matches_df['conversation_started']]
    if len(conv_matches) > 0:
        print(f"  Avg Messages per Conversation: {conv_matches['message_count'].mean():.1f}")
        print(f"  Median Messages: {conv_matches['message_count'].median():.0f}")
        print(f"  Max Messages: {conv_matches['message_count'].max()}")

    # Temporal validation
    print(f"\nTEMPORAL CONSISTENCY:")
    users_df['created_at_dt'] = pd.to_datetime(users_df['created_at'])
    likes_df['timestamp_dt'] = pd.to_datetime(likes_df['timestamp'])

    print(f"  User Join Date Range: {users_df['created_at_dt'].min().date()} to {users_df['created_at_dt'].max().date()}")
    print(f"  Like Date Range: {likes_df['timestamp_dt'].min().date()} to {likes_df['timestamp_dt'].max().date()}")

    # Check for temporal inconsistencies
    user_join_dates = dict(zip(users_df['user_id'], users_df['created_at_dt']))
    temporal_errors = 0

    for _, like in likes_df.iterrows():
        user_joined = user_join_dates[like['user_id']]
        if like['timestamp_dt'] < user_joined:
            temporal_errors += 1

    print(f"  Temporal Errors (likes before join): {temporal_errors}")

    # Duplicate checks
    print(f"\nDATA QUALITY:")
    print(f"  Duplicate Users: {users_df['user_id'].duplicated().sum()}")
    print(f"  Duplicate Likes: {likes_df['like_id'].duplicated().sum()}")
    print(f"  Duplicate Matches: {matches_df['match_id'].duplicated().sum()}")

    print(f"\n{'='*60}\n")


def main():
    """Main execution function."""
    print(f"\n{'='*80}")
    print(f"WORKHEART DATA GENERATION - ULTRA-CONCENTRATED VERSION")
    print(f"{'='*80}\n")

    # Generate users
    users_df = generate_users(NUM_USERS)

    # Generate ultra-concentrated likes with strong CF patterns
    # Using new function for 20-35% user overlap (vs 0.5% in old version)
    likes_df, matches_df = generate_ultra_concentrated_likes(users_df, TARGET_TOTAL_LIKES)

    # Validate and summarize
    validate_and_summarize(users_df, likes_df, matches_df)

    # Save to CSV
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Saving data to CSV files...")

    # Prepare users for export (flatten location and matching preferences)
    users_export = users_df.copy()

    # Drop temporary columns
    if 'planned_likes' in users_export.columns:
        users_export = users_export.drop('planned_likes', axis=1)
    if 'created_at_dt' in users_export.columns:
        users_export = users_export.drop('created_at_dt', axis=1)

    # Save files
    users_export.to_csv('data/raw/users.csv', index=False, encoding='utf-8')
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved data/raw/users.csv")

    likes_export = likes_df.copy()
    if 'timestamp_dt' in likes_export.columns:
        likes_export = likes_export.drop('timestamp_dt', axis=1)

    likes_export.to_csv('data/raw/likes.csv', index=False, encoding='utf-8')
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved data/raw/likes.csv")

    matches_df.to_csv('data/raw/matches.csv', index=False, encoding='utf-8')
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved data/raw/matches.csv")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Data generation complete!\n")


if __name__ == "__main__":
    main()

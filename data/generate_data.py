"""
Generate realistic synthetic data for WorkHeart dating app recommendation system.

This script generates:
- 20,000 users (white-collar professionals, ages 25-45)
- ~250,000 likes (power law distribution over 1-year timeline)
- ~25,000 matches (~10% match rate)

Data simulates 1 year of app operation with realistic user lifecycle patterns.
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

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)
fake = Faker()
Faker.seed(42)

# Configuration
NUM_USERS = 20000
TARGET_TOTAL_LIKES = 250000
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
    """Generate realistic user bio."""
    templates = [
        # Career-focused
        lambda: f"{random.choice(['Love', 'Enjoy', 'Passionate about'])} {random.choice(['hiking', 'traveling', 'cooking', 'photography', 'running', 'yoga'])}. {random.choice(['Foodie', 'Coffee enthusiast', 'Wine lover', 'Craft beer fan'])}. Looking for someone to {random.choice(['explore the city with', 'try new restaurants with', 'go on adventures with', 'share good conversations with'])}.",

        # Hobby-focused
        lambda: f"{random.choice(['Weekend', 'Always'])} {random.choice(['adventurer', 'explorer', 'traveler'])}. Into {random.choice(['fitness', 'wellness', 'mindfulness', 'personal growth'])}. {random.choice(['Dog lover', 'Cat person', 'Animal lover'])}. {random.choice(['Netflix and chill?', 'Up for spontaneous trips', 'Always down for brunch', 'Love a good happy hour'])}.",

        # Balanced
        lambda: f"Work in {random.choice(['tech', 'healthcare', 'finance', 'education'])}. {random.choice(['Love to travel', 'Enjoy the outdoors', 'Foodie at heart', 'Fitness enthusiast'])}. {random.choice(['Looking for meaningful connections', 'Here for something real', 'Quality over quantity', 'No games, just genuine connections'])}.",

        # Short and sweet
        lambda: f"{random.choice(['Adventurer', 'Dreamer', 'Explorer', 'Optimist'])}. {random.choice(['Coffee addict', 'Book lover', 'Music junkie', 'Film buff'])}. {random.choice(['Lets grab drinks', 'Coffee first?', 'Tacos and margaritas?', 'Wine bar regular'])}.",

        # Detailed
        lambda: f"Born and raised in {random.choice(list(CITIES.keys()))}. Love {random.choice(['my job', 'what I do', 'my career'])} but also know how to unplug. You can find me {random.choice(['at the gym', 'trying new restaurants', 'exploring the city', 'at a coffee shop', 'hiking on weekends'])}. {random.choice(['Sapiosexual', 'Looking for my adventure partner', 'Seeking someone who can keep up with me', 'Want someone who makes me laugh'])}.",
    ]

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


def generate_likes(users_df: pd.DataFrame, target_likes: int) -> pd.DataFrame:
    """Generate likes with power law distribution and compatibility-based preferences."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Generating ~{target_likes} likes...")

    # Determine activity level for each user (power law distribution)
    # 5% super active, 15% very active, 30% active, 50% moderate/low
    activity_levels = []

    for idx, user in users_df.iterrows():
        days_active = user['days_since_join']

        # Base likes on user lifecycle
        if days_active >= 270:  # 9-12 months (early adopters)
            base_likes = random.randint(50, 300)
        elif days_active >= 90:  # 3-9 months
            base_likes = random.randint(20, 80)
        else:  # 0-3 months (recent users)
            base_likes = random.randint(5, 30)

        # Apply power law distribution
        rand = random.random()
        if rand < 0.05:  # 5% super active
            likes_to_give = int(base_likes * random.uniform(2.0, 3.0))
        elif rand < 0.20:  # 15% very active
            likes_to_give = int(base_likes * random.uniform(1.2, 2.0))
        elif rand < 0.50:  # 30% active
            likes_to_give = int(base_likes * random.uniform(0.8, 1.2))
        else:  # 50% moderate/low
            likes_to_give = int(base_likes * random.uniform(0.2, 0.8))

        # Inactive users give fewer likes
        if not user['is_active']:
            likes_to_give = int(likes_to_give * random.uniform(0.1, 0.3))

        activity_levels.append(max(1, likes_to_give))

    users_df['planned_likes'] = activity_levels

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Planned total likes: {sum(activity_levels)}")

    # Generate likes
    likes = []
    users_list = users_df.to_dict('records')

    for idx, user in enumerate(users_list):
        if (idx + 1) % 5000 == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing likes for user {idx + 1}/{len(users_list)}...")

        num_likes = activity_levels[idx]

        # Calculate compatibility with all potential matches
        potential_matches = []
        for other_idx, other_user in enumerate(users_list):
            if idx == other_idx:
                continue

            compatibility = calculate_compatibility(user, other_user)
            if compatibility > 0:
                potential_matches.append((other_idx, compatibility))

        if not potential_matches:
            continue

        # Select users to like based on compatibility (weighted random)
        indices, weights = zip(*potential_matches)

        # Don't try to sample more than available
        sample_size = min(num_likes, len(indices))

        liked_indices = random.choices(
            indices,
            weights=weights,
            k=sample_size
        )

        # Generate like records
        user_created_at = datetime.fromisoformat(user['created_at'])
        user_last_login = datetime.fromisoformat(user['last_login'])

        for liked_idx in liked_indices:
            liked_user = users_list[liked_idx]

            # Like timestamp must be after both users joined and before last login
            earliest = max(
                user_created_at,
                datetime.fromisoformat(liked_user['created_at'])
            )

            # Random timestamp between join and last login
            if earliest < user_last_login:
                time_diff = (user_last_login - earliest).total_seconds()
                random_seconds = random.uniform(0, time_diff)
                like_timestamp = earliest + timedelta(seconds=random_seconds)
            else:
                like_timestamp = earliest

            # Action type: 90% like, 10% superlike
            action = 'superlike' if random.random() < 0.10 else 'like'

            likes.append({
                'like_id': str(uuid.uuid4()),
                'user_id': user['user_id'],
                'liked_user_id': liked_user['user_id'],
                'timestamp': like_timestamp.isoformat(),
                'action': action
            })

    likes_df = pd.DataFrame(likes)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Like generation complete! Total: {len(likes_df)}")
    return likes_df


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
    print(f"\n{'='*60}")
    print(f"WORKHEART DATA GENERATION")
    print(f"{'='*60}\n")

    # Generate users
    users_df = generate_users(NUM_USERS)

    # Generate likes
    likes_df = generate_likes(users_df, TARGET_TOTAL_LIKES)

    # Generate matches
    matches_df = generate_matches(users_df, likes_df)

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
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Saved data/raw/users.csv")

    likes_export = likes_df.copy()
    if 'timestamp_dt' in likes_export.columns:
        likes_export = likes_export.drop('timestamp_dt', axis=1)

    likes_export.to_csv('data/raw/likes.csv', index=False, encoding='utf-8')
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Saved data/raw/likes.csv")

    matches_df.to_csv('data/raw/matches.csv', index=False, encoding='utf-8')
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Saved data/raw/matches.csv")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Data generation complete! 🎉\n")


if __name__ == "__main__":
    main()

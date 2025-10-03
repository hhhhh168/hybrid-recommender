"""
Data loader module for WorkHeart recommendation system.

This module handles loading and validating CSV data from the raw data directory.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import warnings

from src.utils import setup_logger, Config

# Initialize logger
logger = setup_logger(__name__)


class DataLoader:
    """
    Load and validate WorkHeart dating app data from CSV files.

    This class handles loading users, likes, and matches data from raw CSV files,
    performs validation, and provides data summary statistics.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize DataLoader.

        Args:
            config: Configuration object for paths (defaults to Config())
        """
        self.config = config if config is not None else Config()
        self.users_df: Optional[pd.DataFrame] = None
        self.likes_df: Optional[pd.DataFrame] = None
        self.matches_df: Optional[pd.DataFrame] = None

        logger.info(f"DataLoader initialized with data directory: {self.config.raw_data_dir}")

    def load_users(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load users data from CSV.

        Args:
            file_path: Optional custom path to users CSV file.
                      Defaults to data/raw/users.csv

        Returns:
            DataFrame with user data and proper dtypes

        Raises:
            FileNotFoundError: If the users CSV file doesn't exist
            ValueError: If required columns are missing
        """
        if file_path is None:
            file_path = self.config.raw_data_dir / "users.csv"
        else:
            file_path = Path(file_path)

        logger.info(f"Loading users data from {file_path}")

        if not file_path.exists():
            raise FileNotFoundError(f"Users file not found: {file_path}")

        try:
            # Load CSV
            df = pd.read_csv(file_path)

            # Validate required columns
            required_cols = [
                'user_id', 'name', 'email', 'gender', 'birthday',
                'city', 'latitude', 'longitude', 'geohash',
                'job_title', 'school', 'created_at', 'last_login'
            ]

            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Parse dates
            df['created_at'] = pd.to_datetime(df['created_at'])
            df['last_login'] = pd.to_datetime(df['last_login'])

            # Calculate age from birthday (MM/DD/YYYY format)
            df['age_calculated'] = df['birthday'].apply(self._calculate_age)

            # Convert boolean columns
            bool_cols = ['email_verified', 'is_premium', 'is_super', 'is_active']
            for col in bool_cols:
                if col in df.columns:
                    df[col] = df[col].astype(bool)

            # Convert numeric columns
            numeric_cols = [
                'hearts', 'swipes_left', 'latitude', 'longitude',
                'matching_pref_age_min', 'matching_pref_age_max', 'age', 'days_since_join'
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Handle matching preference boolean
            if 'matching_pref_use_location' in df.columns:
                df['matching_pref_use_location'] = df['matching_pref_use_location'].astype(bool)

            # Fill missing bios with empty string
            if 'bio' in df.columns:
                df['bio'] = df['bio'].fillna('')

            self.users_df = df

            logger.info(f"Loaded {len(df)} users")
            logger.info(f"Date range: {df['created_at'].min()} to {df['created_at'].max()}")
            logger.info(f"Age range: {df['age'].min():.0f} to {df['age'].max():.0f}")
            logger.info(f"Cities: {df['city'].nunique()} unique cities")

            return df

        except Exception as e:
            logger.error(f"Error loading users data: {e}")
            raise

    def load_likes(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load likes data from CSV.

        Args:
            file_path: Optional custom path to likes CSV file.
                      Defaults to data/raw/likes.csv

        Returns:
            DataFrame with likes data and proper dtypes

        Raises:
            FileNotFoundError: If the likes CSV file doesn't exist
            ValueError: If required columns are missing
        """
        if file_path is None:
            file_path = self.config.raw_data_dir / "likes.csv"
        else:
            file_path = Path(file_path)

        logger.info(f"Loading likes data from {file_path}")

        if not file_path.exists():
            raise FileNotFoundError(f"Likes file not found: {file_path}")

        try:
            # Load CSV
            df = pd.read_csv(file_path)

            # Validate required columns
            required_cols = ['user_id', 'liked_user_id', 'timestamp', 'action']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Parse timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Ensure action is categorical
            df['action'] = df['action'].astype('category')

            # Add like_id if not present
            if 'like_id' not in df.columns:
                df['like_id'] = range(len(df))

            self.likes_df = df

            logger.info(f"Loaded {len(df)} likes")
            logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            logger.info(f"Action distribution: {df['action'].value_counts().to_dict()}")
            logger.info(f"Unique users who liked: {df['user_id'].nunique()}")
            logger.info(f"Unique users who were liked: {df['liked_user_id'].nunique()}")

            return df

        except Exception as e:
            logger.error(f"Error loading likes data: {e}")
            raise

    def load_matches(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load matches data from CSV.

        Args:
            file_path: Optional custom path to matches CSV file.
                      Defaults to data/raw/matches.csv

        Returns:
            DataFrame with matches data and proper dtypes

        Raises:
            FileNotFoundError: If the matches CSV file doesn't exist
            ValueError: If required columns are missing
        """
        if file_path is None:
            file_path = self.config.raw_data_dir / "matches.csv"
        else:
            file_path = Path(file_path)

        logger.info(f"Loading matches data from {file_path}")

        if not file_path.exists():
            raise FileNotFoundError(f"Matches file not found: {file_path}")

        try:
            # Load CSV
            df = pd.read_csv(file_path)

            # Validate required columns
            required_cols = ['match_id', 'user1_id', 'user2_id', 'matched_at']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Parse timestamps
            df['matched_at'] = pd.to_datetime(df['matched_at'])

            # Parse last_message_at (may have NaNs)
            if 'last_message_at' in df.columns:
                df['last_message_at'] = pd.to_datetime(df['last_message_at'], errors='coerce')

            # Convert boolean
            if 'conversation_started' in df.columns:
                df['conversation_started'] = df['conversation_started'].astype(bool)

            # Convert numeric
            if 'message_count' in df.columns:
                df['message_count'] = pd.to_numeric(df['message_count'], errors='coerce').fillna(0).astype(int)

            self.matches_df = df

            logger.info(f"Loaded {len(df)} matches")
            logger.info(f"Date range: {df['matched_at'].min()} to {df['matched_at'].max()}")
            if 'conversation_started' in df.columns:
                conv_count = df['conversation_started'].sum()
                logger.info(f"Matches with conversations: {conv_count} ({conv_count/len(df)*100:.1f}%)")
            if 'message_count' in df.columns:
                logger.info(f"Avg messages per match: {df['message_count'].mean():.1f}")

            return df

        except Exception as e:
            logger.error(f"Error loading matches data: {e}")
            raise

    def load_all(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all data files (users, likes, matches).

        Returns:
            Tuple of (users_df, likes_df, matches_df)

        Raises:
            FileNotFoundError: If any required file doesn't exist
        """
        logger.info("Loading all data files...")

        users_df = self.load_users()
        likes_df = self.load_likes()
        matches_df = self.load_matches()

        logger.info("All data files loaded successfully")

        return users_df, likes_df, matches_df

    def validate_data(self) -> bool:
        """
        Validate loaded data for consistency and completeness.

        Returns:
            True if all validations pass, False otherwise

        Raises:
            ValueError: If data hasn't been loaded yet
        """
        if self.users_df is None or self.likes_df is None or self.matches_df is None:
            raise ValueError("Data must be loaded before validation. Call load_all() first.")

        logger.info("Validating data...")

        all_valid = True

        # Check for critical null values in users
        user_critical_cols = ['user_id', 'gender', 'birthday', 'city', 'created_at']
        for col in user_critical_cols:
            null_count = self.users_df[col].isnull().sum()
            if null_count > 0:
                logger.warning(f"Users: {null_count} null values in critical column '{col}'")
                all_valid = False

        # Check for duplicates
        user_dupes = self.users_df['user_id'].duplicated().sum()
        if user_dupes > 0:
            logger.warning(f"Users: {user_dupes} duplicate user_ids found")
            all_valid = False

        # Check for critical null values in likes
        like_critical_cols = ['user_id', 'liked_user_id', 'timestamp']
        for col in like_critical_cols:
            null_count = self.likes_df[col].isnull().sum()
            if null_count > 0:
                logger.warning(f"Likes: {null_count} null values in critical column '{col}'")
                all_valid = False

        # Validate user references in likes
        user_ids = set(self.users_df['user_id'])
        invalid_likers = ~self.likes_df['user_id'].isin(user_ids)
        invalid_liked = ~self.likes_df['liked_user_id'].isin(user_ids)

        if invalid_likers.sum() > 0:
            logger.warning(f"Likes: {invalid_likers.sum()} likes from unknown users")
            all_valid = False

        if invalid_liked.sum() > 0:
            logger.warning(f"Likes: {invalid_liked.sum()} likes to unknown users")
            all_valid = False

        # Validate user references in matches
        invalid_user1 = ~self.matches_df['user1_id'].isin(user_ids)
        invalid_user2 = ~self.matches_df['user2_id'].isin(user_ids)

        if invalid_user1.sum() > 0:
            logger.warning(f"Matches: {invalid_user1.sum()} matches with unknown user1_id")
            all_valid = False

        if invalid_user2.sum() > 0:
            logger.warning(f"Matches: {invalid_user2.sum()} matches with unknown user2_id")
            all_valid = False

        # Temporal validation: likes should be after user creation
        user_join_dates = dict(zip(self.users_df['user_id'], self.users_df['created_at']))

        temporal_errors = 0
        for _, like in self.likes_df.iterrows():
            user_joined = user_join_dates.get(like['user_id'])
            if user_joined and like['timestamp'] < user_joined:
                temporal_errors += 1

        if temporal_errors > 0:
            logger.warning(f"Temporal: {temporal_errors} likes before user join date")
            all_valid = False

        if all_valid:
            logger.info("✓ All validation checks passed")
        else:
            logger.warning("✗ Some validation checks failed")

        return all_valid

    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics about loaded data.

        Returns:
            Dictionary with summary statistics

        Raises:
            ValueError: If data hasn't been loaded yet
        """
        if self.users_df is None or self.likes_df is None or self.matches_df is None:
            raise ValueError("Data must be loaded before getting summary. Call load_all() first.")

        summary = {
            'users': {
                'total': len(self.users_df),
                'gender_distribution': self.users_df['gender'].value_counts().to_dict(),
                'age_range': (int(self.users_df['age'].min()), int(self.users_df['age'].max())),
                'avg_age': float(self.users_df['age'].mean()),
                'premium_users': int(self.users_df['is_premium'].sum()) if 'is_premium' in self.users_df.columns else 0,
                'active_users': int(self.users_df['is_active'].sum()) if 'is_active' in self.users_df.columns else 0,
                'cities': self.users_df['city'].nunique(),
                'date_range': (
                    str(self.users_df['created_at'].min()),
                    str(self.users_df['created_at'].max())
                )
            },
            'likes': {
                'total': len(self.likes_df),
                'unique_likers': self.likes_df['user_id'].nunique(),
                'unique_liked': self.likes_df['liked_user_id'].nunique(),
                'avg_likes_per_user': float(len(self.likes_df) / self.likes_df['user_id'].nunique()),
                'action_distribution': self.likes_df['action'].value_counts().to_dict(),
                'date_range': (
                    str(self.likes_df['timestamp'].min()),
                    str(self.likes_df['timestamp'].max())
                )
            },
            'matches': {
                'total': len(self.matches_df),
                'with_conversation': int(self.matches_df['conversation_started'].sum()) if 'conversation_started' in self.matches_df.columns else 0,
                'avg_messages': float(self.matches_df['message_count'].mean()) if 'message_count' in self.matches_df.columns else 0,
                'match_rate': float(len(self.matches_df) * 2 / len(self.likes_df) * 100) if len(self.likes_df) > 0 else 0,
                'date_range': (
                    str(self.matches_df['matched_at'].min()),
                    str(self.matches_df['matched_at'].max())
                )
            }
        }

        return summary

    @staticmethod
    def _calculate_age(birthday_str: str) -> int:
        """
        Calculate age from birthday string in MM/DD/YYYY format.

        Args:
            birthday_str: Birthday in MM/DD/YYYY format

        Returns:
            Age in years
        """
        try:
            birth_date = datetime.strptime(birthday_str, '%m/%d/%Y')
            today = datetime.now()
            age = today.year - birth_date.year - (
                (today.month, today.day) < (birth_date.month, birth_date.day)
            )
            return age
        except (ValueError, AttributeError):
            logger.warning(f"Could not parse birthday: {birthday_str}")
            return 0

"""
Feature engineering module for Airbnb Tokyo dataset.
Creates new features for modeling and analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Create features from cleaned Airbnb data."""

    def engineer_listings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features for listings data.

        Args:
            df: Cleaned listings dataframe

        Returns:
            Dataframe with new features
        """
        logger.info("Engineering listing features...")
        df_feat = df.copy()

        # 1. Price per bedroom
        if 'price' in df_feat.columns and 'bedrooms' in df_feat.columns:
            df_feat['price_per_bedroom'] = df_feat['price'] / df_feat['bedrooms'].replace(0, 1)

        # 2. Price per accommodation
        if 'price' in df_feat.columns and 'accommodates' in df_feat.columns:
            df_feat['price_per_guest'] = df_feat['price'] / df_feat['accommodates'].replace(0, 1)

        # 3. Host experience
        if 'host_since' in df_feat.columns:
            df_feat['host_days_active'] = (pd.Timestamp.now() - df_feat['host_since']).dt.days
            df_feat['host_years_active'] = df_feat['host_days_active'] / 365.25

        # 4. Review popularity
        if 'number_of_reviews' in df_feat.columns and 'reviews_per_month' in df_feat.columns:
            df_feat['is_popular'] = df_feat['number_of_reviews'] > df_feat['number_of_reviews'].median()
            df_feat['review_velocity'] = df_feat['reviews_per_month'].fillna(0)

        # 5. Listing score (composite)
        if all(col in df_feat.columns for col in ['review_scores_rating', 'availability_365']):
            # Normalize and combine
            rating_norm = df_feat['review_scores_rating'] / 100
            availability_norm = 1 - (df_feat['availability_365'] / 365)
            df_feat['listing_score'] = (rating_norm * 0.7 + availability_norm * 0.3) * 100

        # 6. Response rate category
        if 'host_response_rate' in df_feat.columns:
            df_feat['response_rate_numeric'] = df_feat['host_response_rate'].str.replace('%', '').astype(float)
            df_feat['response_category'] = pd.cut(
                df_feat['response_rate_numeric'],
                bins=[0, 50, 80, 90, 100],
                labels=['Low', 'Medium', 'High', 'Excellent']
            )

        # 7. Instant bookable
        if 'instant_bookable' in df_feat.columns:
            df_feat['instant_book'] = df_feat['instant_bookable'].map({'t': 1, 'f': 0})

        # 8. Superhost
        if 'host_is_superhost' in df_feat.columns:
            df_feat['is_superhost'] = df_feat['host_is_superhost'].map({'t': 1, 'f': 0})

        # 9. Price category (for easy filtering)
        if 'price' in df_feat.columns:
            price_percentiles = df_feat['price'].quantile([0.25, 0.50, 0.75])
            df_feat['price_category'] = pd.cut(
                df_feat['price'],
                bins=[0, price_percentiles[0.25], price_percentiles[0.50],
                      price_percentiles[0.75], float('inf')],
                labels=['Budget', 'Moderate', 'Premium', 'Luxury']
            )

        # 10. Density features (will be filled later with geospatial)
        df_feat['neighborhood_count'] = df_feat.groupby('neighbourhood_cleansed')['id'].transform('count') \
            if 'neighbourhood_cleansed' in df_feat.columns else 0

        logger.info(f"Created {len(df_feat.columns)} features")
        return df_feat

    def engineer_calendar(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from calendar data.

        Args:
            df: Cleaned calendar dataframe

        Returns:
            Dataframe with time-series features
        """
        logger.info("Engineering calendar features...")
        df_feat = df.copy()

        if 'date' in df_feat.columns:
            # Time-based features
            df_feat['year'] = df_feat['date'].dt.year
            df_feat['month'] = df_feat['date'].dt.month
            df_feat['day'] = df_feat['date'].dt.day
            df_feat['day_of_week'] = df_feat['date'].dt.dayofweek
            df_feat['quarter'] = df_feat['date'].dt.quarter
            df_feat['is_weekend'] = df_feat['day_of_week'].isin([5, 6]).astype(int)

            # Season
            df_feat['season'] = df_feat['month'].map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Fall', 10: 'Fall', 11: 'Fall'
            })

        logger.info(f"Calendar features engineered")
        return df_feat

    def engineer_all(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Engineer features for all dataframes.

        Args:
            data: Dictionary of cleaned dataframes

        Returns:
            Dictionary with engineered features
        """
        logger.info("=" * 50)
        logger.info("Engineering features for all data...")
        logger.info("=" * 50)

        engineered = {}

        if 'listings' in data:
            engineered['listings'] = self.engineer_listings(data['listings'])

        if 'calendar' in data:
            engineered['calendar'] = self.engineer_calendar(data['calendar'])

        # Pass through unchanged
        for key in ['reviews', 'neighbourhoods', 'neighbourhoods_geojson']:
            if key in data:
                engineered[key] = data[key]

        logger.info("=" * 50)
        logger.info("Feature engineering complete!")
        logger.info("=" * 50)

        return engineered
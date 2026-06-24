"""
Data cleaning module for Airbnb Tokyo dataset.
Handles missing values, data type conversions, and price cleaning.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCleaner:
    """Clean and preprocess Airbnb data."""

    def clean_price(self, price_series: pd.Series) -> pd.Series:
        """
        Clean price column by removing $ and converting to float.
        Handles both string and numeric inputs.
        
        Args:
            price_series: Series with price values
            
        Returns:
            Series with float prices
        """
        # Convert to string first to handle any data type
        cleaned = price_series.astype(str)
        
        # Remove $, commas, and any other non-numeric characters except decimal
        cleaned = cleaned.str.replace('$', '', regex=False)
        cleaned = cleaned.str.replace(',', '', regex=False)
        cleaned = cleaned.str.replace('¥', '', regex=False)  # Handle yen symbol if present
        cleaned = cleaned.str.replace(' ', '', regex=False)  # Remove spaces
        
        # Remove any trailing text like "per night"
        cleaned = cleaned.str.replace(r'per night.*$', '', regex=True)
        
        # Convert to numeric, coerce errors to NaN
        result = pd.to_numeric(cleaned, errors='coerce')
        
        # Log if all values are NaN
        if result.isna().all():
            logger.warning("⚠️ All prices are NaN after cleaning. Check raw data format.")
        
        return result

    def clean_listings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the listings dataframe.

        Args:
            df: Raw listings dataframe

        Returns:
            Cleaned listings dataframe
        """
        logger.info("Cleaning listings data...")
        df_clean = df.copy()

        # 1. Clean price columns
        if 'price' in df_clean.columns:
            df_clean['price'] = self.clean_price(df_clean['price'])
            logger.info(f"Cleaned price column")

        # 2. Clean other price-like columns
        price_cols = ['weekly_price', 'monthly_price', 'security_deposit',
                      'cleaning_fee', 'extra_people']
        for col in price_cols:
            if col in df_clean.columns:
                df_clean[col] = self.clean_price(df_clean[col])

        # 3. Handle missing values for key columns
        # For numeric columns, fill with median
        numeric_cols = ['bathrooms', 'bedrooms', 'beds', 'review_scores_rating',
                        'reviews_per_month']
        for col in numeric_cols:
            if col in df_clean.columns:
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                logger.info(f"Filled missing {col} with median: {median_val:.2f}")

        # 4. Convert date columns
        date_cols = ['last_scraped', 'first_review', 'last_review', 'host_since']
        for col in date_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')

        # 5. Create derived columns
        if 'review_scores_rating' in df_clean.columns:
            df_clean['has_reviews'] = df_clean['review_scores_rating'].notna()
            df_clean['rating_category'] = pd.cut(
                df_clean['review_scores_rating'],
                bins=[0, 80, 90, 95, 100],
                labels=['Poor', 'Good', 'Great', 'Excellent']
            )

        # 6. Drop columns with too many missing values (>50%)
        threshold = 0.5 * len(df_clean)
        cols_to_drop = [col for col in df_clean.columns
                        if df_clean[col].isna().sum() > threshold]
        if cols_to_drop:
            df_clean.drop(columns=cols_to_drop, inplace=True)
            logger.info(f"Dropped {len(cols_to_drop)} columns with >50% missing")

        logger.info(f"Listings cleaned: {len(df_clean):,} rows, {len(df_clean.columns)} columns")
        return df_clean

    def clean_calendar(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the calendar dataframe.
        
        Args:
            df: Raw calendar dataframe
            
        Returns:
            Cleaned calendar dataframe
        """
        logger.info("Cleaning calendar data...")
        df_clean = df.copy()

        # Convert date to datetime
        if 'date' in df_clean.columns:
            df_clean['date'] = pd.to_datetime(df_clean['date'])

        # Clean price columns - use the improved clean_price method
        if 'price' in df_clean.columns:
            # First, check what the raw data looks like
            raw_sample = df_clean['price'].head(10)
            logger.info(f"   Raw price sample: {raw_sample.tolist()}")
            
            # Clean the prices
            df_clean['price'] = self.clean_price(df_clean['price'])
            
            # Log results
            non_null = df_clean['price'].count()
            logger.info(f"   Price column cleaned: {non_null:,} non-null values")
            
            if non_null == 0:
                logger.error("⚠️ Still no valid prices! Please check raw data format.")

        if 'adjusted_price' in df_clean.columns:
            df_clean['adjusted_price'] = self.clean_price(df_clean['adjusted_price'])

        # Convert available to boolean
        if 'available' in df_clean.columns:
            df_clean['available'] = df_clean['available'].map({'t': True, 'f': False})

        logger.info(f"✅ Calendar cleaned: {len(df_clean):,} rows")
        return df_clean

    def clean_reviews(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the reviews dataframe.

        Args:
            df: Raw reviews dataframe

        Returns:
            Cleaned reviews dataframe
        """
        logger.info("Cleaning reviews data...")
        df_clean = df.copy()

        # Convert date to datetime
        if 'date' in df_clean.columns:
            df_clean['date'] = pd.to_datetime(df_clean['date'])

        # Fill missing reviewer names
        if 'reviewer_name' in df_clean.columns:
            df_clean['reviewer_name'].fillna('Anonymous', inplace=True)

        # Fill missing comments
        if 'comments' in df_clean.columns:
            df_clean['comments'].fillna('', inplace=True)

        # Add review length feature
        if 'comments' in df_clean.columns:
            df_clean['review_length'] = df_clean['comments'].str.len()

        logger.info(f"Reviews cleaned: {len(df_clean):,} rows")
        return df_clean

    def clean_all(self, data: Dict[str, pd.DataFrame], 
              sample_calendar: bool = False,
              sample_reviews: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Clean all dataframes.
        
        Args:
            data: Dictionary of raw dataframes
            sample_calendar: If True, sample calendar to 100k rows (default: False)
            sample_reviews: If True, sample reviews to 50k rows (default: False)
        
        Returns:
            Dictionary of cleaned dataframes
        """
        logger.info("=" * 50)
        logger.info("Cleaning all data...")
        logger.info("=" * 50)

        cleaned = {}

        if 'listings' in data:
            cleaned['listings'] = self.clean_listings(data['listings'])

        if 'calendar' in data:
            if sample_calendar:
                calendar_sample = data['calendar'].sample(n=min(100000, len(data['calendar'])), random_state=42)
                cleaned['calendar'] = self.clean_calendar(calendar_sample)
                logger.info(f"💡 Calendar sampled to {len(cleaned['calendar']):,} rows")
            else:
                cleaned['calendar'] = self.clean_calendar(data['calendar'])
                logger.info(f"✅ Calendar kept all {len(cleaned['calendar']):,} rows")

        if 'reviews' in data:
            if sample_reviews:
                reviews_sample = data['reviews'].sample(n=min(50000, len(data['reviews'])), random_state=42)
                cleaned['reviews'] = self.clean_reviews(reviews_sample)
                logger.info(f"💡 Reviews sampled to {len(cleaned['reviews']):,} rows")
            else:
                cleaned['reviews'] = self.clean_reviews(data['reviews'])
                logger.info(f"✅ Reviews kept all {len(cleaned['reviews']):,} rows")

        # Pass through unchanged
        if 'neighbourhoods' in data:
            cleaned['neighbourhoods'] = data['neighbourhoods']

        if 'neighbourhoods_geojson' in data:
            cleaned['neighbourhoods_geojson'] = data['neighbourhoods_geojson']

        logger.info("=" * 50)
        logger.info("✅ All data cleaned!")
        logger.info("=" * 50)

        return cleaned
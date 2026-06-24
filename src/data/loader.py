"""
Data loader module for Airbnb Tokyo dataset.
Loads all CSV and GeoJSON files from the raw data directory.
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and validate Airbnb Tokyo data files."""

    def __init__(self, raw_data_path: str = "data/raw"):
        """
        Initialize the DataLoader.

        Args:
            raw_data_path: Path to the directory containing raw data files
        """
        self.raw_data_path = Path(raw_data_path)
        self.data = {}

    def load_listings(self, sample: Optional[int] = None) -> pd.DataFrame:
        """
        Load listings.csv.

        Args:
            sample: Number of rows to sample (optional)

        Returns:
            DataFrame with listing data
        """
        file_path = self.raw_data_path / "listings.csv"
        logger.info(f"Loading listings from {file_path}...")

        df = pd.read_csv(file_path)

        if sample:
            df = df.sample(n=min(sample, len(df)), random_state=42)

        logger.info(f"Loaded {len(df):,} listings")
        return df

    def load_calendar(self, sample: Optional[int] = None) -> pd.DataFrame:
        """
        Load calendar.csv.

        Args:
            sample: Number of rows to sample (optional)

        Returns:
            DataFrame with calendar data
        """
        file_path = self.raw_data_path / "calendar.csv"
        logger.info(f"Loading calendar from {file_path}...")

        # Use low_memory=False to avoid mixed type warnings
        df = pd.read_csv(file_path, low_memory=False)

        if sample:
            df = df.sample(n=min(sample, len(df)), random_state=42)

        logger.info(f"Loaded {len(df):,} calendar entries")
        return df

    def load_reviews(self, sample: Optional[int] = None) -> pd.DataFrame:
        """
        Load reviews.csv.

        Args:
            sample: Number of rows to sample (optional)

        Returns:
            DataFrame with review data
        """
        file_path = self.raw_data_path / "reviews.csv"
        logger.info(f"Loading reviews from {file_path}...")

        df = pd.read_csv(file_path)

        if sample:
            df = df.sample(n=min(sample, len(df)), random_state=42)

        logger.info(f"Loaded {len(df):,} reviews")
        return df

    def load_reviews_recent(self, years: int = 2, sample: Optional[int] = None) -> pd.DataFrame:
        """
        Load reviews from the last N years only.
        
        Args:
            years: Number of recent years to keep (default: 2)
            sample: Number of rows to sample (optional)
        
        Returns:
            DataFrame with recent review data
        """
        file_path = self.raw_data_path / "reviews.csv"
        logger.info(f"Loading recent {years} years of reviews from {file_path}...")
        
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate cutoff date
        cutoff_date = pd.Timestamp.now() - pd.DateOffset(years=years)
        
        # Filter to recent reviews
        df = df[df['date'] >= cutoff_date]
        logger.info(f"Filtered to reviews after {cutoff_date.strftime('%Y-%m-%d')}: {len(df):,} rows")
        
        if sample:
            df = df.sample(n=min(sample, len(df)), random_state=42)
        
        logger.info(f"✅ Loaded {len(df):,} recent reviews")
        return df

    def load_neighbourhoods(self) -> pd.DataFrame:
        """Load neighbourhoods.csv."""
        file_path = self.raw_data_path / "neighbourhoods.csv"
        logger.info(f"Loading neighbourhoods from {file_path}...")

        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} neighbourhoods")
        return df

    def load_neighbourhoods_geojson(self) -> gpd.GeoDataFrame:
        """Load neighbourhoods.geojson."""
        file_path = self.raw_data_path / "neighbourhoods.geojson"
        logger.info(f"Loading GeoJSON from {file_path}...")

        gdf = gpd.read_file(file_path)
        logger.info(f"Loaded {len(gdf)} neighbourhood geometries")
        return gdf

    def load_all(self, sample_listings: Optional[int] = None,
             sample_calendar: Optional[int] = None,
             sample_reviews: Optional[int] = None,
             recent_reviews_years: int = 2) -> Dict[str, pd.DataFrame]:
        """
        Load all data files.
        
        Args:
            sample_listings: Sample size for listings
            sample_calendar: Sample size for calendar
            sample_reviews: Sample size for reviews
            recent_reviews_years: Number of recent years for reviews (default: 2)
        
        Returns:
            Dictionary with all dataframes
        """
        logger.info("=" * 50)
        logger.info("Loading all data files...")
        logger.info("=" * 50)
        
        self.data = {
            'listings': self.load_listings(sample_listings),
            'calendar': self.load_calendar(sample_calendar),
            'reviews': self.load_reviews_recent(recent_reviews_years, sample_reviews),
            'neighbourhoods': self.load_neighbourhoods(),
            'neighbourhoods_geojson': self.load_neighbourhoods_geojson()
        }
        
        logger.info("=" * 50)
        logger.info("✅ All data loaded successfully!")
        logger.info("=" * 50)
        
        return self.data

    def get_summary(self) -> pd.DataFrame:
        """Get summary of all loaded datasets."""
        if not self.data:
            raise ValueError("No data loaded. Call load_all() first.")

        summary = []
        for name, df in self.data.items():
            summary.append({
                'Dataset': name,
                'Rows': f"{len(df):,}",
                'Columns': len(df.columns),
                'Memory (MB)': f"{df.memory_usage(deep=True).sum() / 1024**2:.1f}"
            })

        return pd.DataFrame(summary)
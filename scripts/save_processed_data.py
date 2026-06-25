"""
Save processed data to parquet files for fast loading.
"""

import sys
sys.path.append('.')

from src.data.loader import DataLoader
from src.data.cleaner import DataCleaner
from src.data.feature_engineer import FeatureEngineer
import pandas as pd
from pathlib import Path

def main():
    print("=" * 60)
    print("SAVING PROCESSED DATA")
    print("=" * 60)

    # 1. Load data with 2-year reviews filter
    print("\nLoading data...")
    loader = DataLoader()
    raw_data = loader.load_all(
        sample_listings=None,
        sample_calendar=None,  # Keep full calendar for forecasting
        sample_reviews=None,   # Keep all recent reviews (444k)
        recent_reviews_years=2
    )

    # 2. Clean data
    print("\nCleaning data (keeping ALL rows)...")
    cleaner = DataCleaner()
    cleaned_data = cleaner.clean_all(
        raw_data,
        sample_calendar=False,  # Keep ALL calendar data
        sample_reviews=False    # Keep ALL reviews data
    )

    # 3. Engineer features
    print("\nEngineering features...")
    engineer = FeatureEngineer()
    final_data = engineer.engineer_all(cleaned_data)

    # 4. Create processed directory
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    # 5. Save each dataset
    print("\nSaving to data/processed/...")

    final_data['listings'].to_parquet('data/processed/listings_processed.parquet')
    print(f"  listings_processed.parquet ({len(final_data['listings']):,} rows)")

    final_data['calendar'].to_parquet('data/processed/calendar_processed.parquet')
    print(f"  calendar_processed.parquet ({len(final_data['calendar']):,} rows)")

    final_data['reviews'].to_parquet('data/processed/reviews_processed.parquet')
    print(f"  reviews_processed.parquet ({len(final_data['reviews']):,} rows)")

    # 6. Save a small sample for quick testing
    final_data['listings'].sample(1000, random_state=42).to_parquet('data/processed/listings_sample_1000.parquet')
    print(f"  listings_sample_1000.parquet (1,000 rows for testing)")

    print("\n" + "=" * 60)
    print("All data saved successfully.")
    print("=" * 60)

    # 7. Show summary
    print("\nDATASET SUMMARY:")
    print(f"  Listings: {len(final_data['listings']):,} rows, {len(final_data['listings'].columns)} columns")
    print(f"  Calendar: {len(final_data['calendar']):,} rows, {len(final_data['calendar'].columns)} columns")
    print(f"  Reviews:  {len(final_data['reviews']):,} rows, {len(final_data['reviews'].columns)} columns")

    # 8. Memory usage
    total_memory = (
        final_data['listings'].memory_usage(deep=True).sum() +
        final_data['calendar'].memory_usage(deep=True).sum() +
        final_data['reviews'].memory_usage(deep=True).sum()
    ) / 1024**3
    print(f"\nTotal memory: {total_memory:.2f} GB")

if __name__ == "__main__":
    main()

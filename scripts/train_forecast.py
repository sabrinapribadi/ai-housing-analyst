"""
Train Prophet forecasting models for each neighborhood.
"""

import sys
sys.path.append('.')

import pandas as pd
from src.models.forecast import PriceForecaster
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    print("=" * 60)
    print("TRAINING FORECASTING MODELS")
    print("=" * 60)

    # 1. Load processed calendar data
    print("\nLoading calendar data...")
    calendar_df = pd.read_parquet("data/processed/calendar_processed.parquet")
    print(f"   Loaded {len(calendar_df):,} calendar entries")

    # 2. Initialize forecaster
    forecaster = PriceForecaster(calendar_df)

    # 3. Train models for top neighborhoods
    listings = pd.read_parquet("data/processed/listings_processed.parquet")
    top_neighborhoods = listings['neighbourhood_cleansed'].value_counts().head(5).index.tolist()

    print(f"\nTraining models for top 5 neighborhoods:")
    for nb in top_neighborhoods:
        print(f"   - {nb}")

    results = {}

    # 4. Train full Tokyo model
    print("\n" + "=" * 60)
    logger.info("Training Full Tokyo Model...")
    results['all_tokyo'] = forecaster.train(forecast_days=90)

    # 5. Train neighborhood models
    for neighborhood in top_neighborhoods:
        print("\n" + "=" * 60)
        logger.info(f"Training Model for: {neighborhood}")
        results[neighborhood] = forecaster.train(neighborhood, forecast_days=90)

    # 6. Save summary
    print("\n" + "=" * 60)
    print("FORECAST SUMMARY")
    print("=" * 60)

    for name, result in results.items():
        metrics = result['metrics']
        summary = forecaster.get_forecast_summary(result['forecast'])
        print(f"\n{name}:")
        print(f"   Training days: {metrics['training_days']:,}")
        print(f"   Avg forecast price: ¥{summary['avg_forecast_price']:,.0f}")
        print(f"   Trend: {summary['trend_direction'].upper()}")
        print(f"   Price range: ¥{summary['price_range_lower']:,.0f} - ¥{summary['price_range_upper']:,.0f}")

    # 7. Save forecast summary
    summary_data = []
    for name, result in results.items():
        metrics = result['metrics']
        summary = forecaster.get_forecast_summary(result['forecast'])
        summary_data.append({
            'neighborhood': name,
            'training_days': metrics['training_days'],
            'avg_price': summary['avg_forecast_price'],
            'trend': summary['trend_direction'],
            'price_lower': summary['price_range_lower'],
            'price_upper': summary['price_range_upper']
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv("data/processed/forecast_summary.csv", index=False)
    print("\nForecast summary saved to data/processed/forecast_summary.csv")
    print("Models saved to models/")
    print("Done.")


if __name__ == "__main__":
    main()

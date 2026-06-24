"""
Time-series forecasting module using Prophet.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from typing import Dict, Optional, Tuple
import logging
from pathlib import Path
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PriceForecaster:
    """Forecast Airbnb prices using Prophet."""

    def __init__(self, calendar_data: Optional[pd.DataFrame] = None):
        """
        Initialize the forecaster.

        Args:
            calendar_data: Cleaned calendar dataframe
        """
        self.calendar_data = calendar_data
        self.model = None
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)

    def prepare_data(self, neighborhood: Optional[str] = None) -> pd.DataFrame:
        """
        Prepare data for Prophet forecasting.

        Args:
            neighborhood: Optional neighborhood to filter by

        Returns:
            DataFrame with 'ds' (date) and 'y' (price) columns
        """
        if self.calendar_data is None:
            raise ValueError("No calendar data loaded")

        # Filter by neighborhood if specified
        df = self.calendar_data.copy()
        if neighborhood:
            # Get listing IDs for this neighborhood
            listings = pd.read_parquet("data/processed/listings_processed.parquet")
            listing_ids = listings[listings['neighbourhood_cleansed'] == neighborhood]['id'].tolist()
            df = df[df['listing_id'].isin(listing_ids)]

        # Aggregate by date (mean price)
        daily_avg = df.groupby('date')['price'].mean().reset_index()
        daily_avg.columns = ['ds', 'y']

        # Remove outliers
        upper_limit = daily_avg['y'].quantile(0.95)
        daily_avg = daily_avg[daily_avg['y'] <= upper_limit]

        logger.info(f"✅ Prepared {len(daily_avg):,} daily records for forecasting")
        if neighborhood:
            logger.info(f"   Neighborhood: {neighborhood}")
        else:
            logger.info(f"   All Tokyo (aggregated)")

        return daily_avg

    def train(self, neighborhood: Optional[str] = None,
              forecast_days: int = 90) -> Dict:
        """
        Train Prophet model and generate forecast.

        Args:
            neighborhood: Optional neighborhood to filter by
            forecast_days: Number of days to forecast

        Returns:
            Dictionary with model, forecast, and metrics
        """
        logger.info("=" * 60)
        logger.info(f"📈 Training Prophet model for: {neighborhood or 'All Tokyo'}")
        logger.info("=" * 60)

        # Prepare data
        df = self.prepare_data(neighborhood)

        # Initialize Prophet
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05,
            interval_width=0.95
        )

        # Add monthly seasonality
        self.model.add_seasonality(
            name='monthly',
            period=30.5,
            fourier_order=5
        )

        # Fit model
        logger.info("🔄 Training model...")
        self.model.fit(df)

        # Create future dataframe
        future = self.model.make_future_dataframe(periods=forecast_days)
        forecast = self.model.predict(future)

        # Extract metrics
        metrics = {
            'neighborhood': neighborhood or 'All Tokyo',
            'training_days': len(df),
            'forecast_days': forecast_days,
            'date_range': f"{df['ds'].min()} to {df['ds'].max()}",
        }

        logger.info(f"✅ Model trained successfully!")
        logger.info(f"   Training days: {len(df):,}")
        logger.info(f"   Forecast days: {forecast_days}")

        # Save model
        model_name = f"prophet_{neighborhood or 'tokyo'}.pkl"
        self.save_model(model_name)

        return {
            'model': self.model,
            'forecast': forecast,
            'metrics': metrics,
            'training_data': df
        }

    def save_model(self, filename: str = "prophet_model.pkl"):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save. Train first.")

        model_path = self.models_dir / filename
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"💾 Model saved to {model_path}")

    def load_model(self, filename: str = "prophet_model.pkl"):
        """Load a saved model."""
        model_path = self.models_dir / filename
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        logger.info(f"📂 Model loaded from {model_path}")

    def forecast_for_neighborhood(self, neighborhood: str,
                                  days: int = 90) -> pd.DataFrame:
        """
        Generate forecast for a specific neighborhood.

        Args:
            neighborhood: Name of the neighborhood
            days: Number of days to forecast

        Returns:
            DataFrame with forecast
        """
        logger.info(f"🔮 Forecasting for {neighborhood}...")
        result = self.train(neighborhood, days)
        return result['forecast'][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    def get_forecast_summary(self, forecast: pd.DataFrame) -> Dict:
        """
        Get summary statistics from forecast.

        Args:
            forecast: Forecast DataFrame from Prophet

        Returns:
            Dictionary with summary metrics
        """
        recent = forecast.tail(90)  # Last 90 days

        return {
            'avg_forecast_price': recent['yhat'].mean(),
            'min_forecast_price': recent['yhat'].min(),
            'max_forecast_price': recent['yhat'].max(),
            'trend_direction': 'up' if recent['yhat'].iloc[-1] > recent['yhat'].iloc[0] else 'down',
            'price_range_lower': recent['yhat_lower'].mean(),
            'price_range_upper': recent['yhat_upper'].mean(),
        }
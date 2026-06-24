"""
EDA automation module for Airbnb Tokyo data.
Provides reusable visualization functions for the LLM agent.
"""

import pandas as pd
import numpy as np
import matplotlib
# Use Agg backend for thread-safe plotting
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EDAAnalyzer:
    """Automated EDA functions for the LLM agent."""

    def __init__(self, listings_df: pd.DataFrame, reviews_df: Optional[pd.DataFrame] = None):
        """
        Initialize with cleaned data.
        
        Args:
            listings_df: Cleaned listings dataframe
            reviews_df: Cleaned reviews dataframe (optional)
        """
        self.listings = listings_df
        self.reviews = reviews_df
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)

    def plot_price_distribution(self, save: bool = True) -> str:
        """Generate price distribution plot."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Full distribution
        self.listings['price'].hist(bins=50, edgecolor='black', ax=axes[0])
        axes[0].set_title('Price Distribution (All)')
        axes[0].set_xlabel('Price (JPY)')
        axes[0].set_ylabel('Frequency')
        
        # Focus on typical range (< 100,000 JPY)
        df_filtered = self.listings[self.listings['price'] < 100000]
        df_filtered['price'].hist(bins=50, edgecolor='black', ax=axes[1])
        axes[1].set_title('Price Distribution (0-100,000 JPY)')
        axes[1].set_xlabel('Price (JPY)')
        axes[1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / 'price_distribution.png'
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"💾 Saved to {path}")
            return str(path)

        plt.show()
        return "displayed"

    def plot_price_by_room_type(self, save: bool = True) -> str:
        """Plot price distribution by room type."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Boxplot (filter to reasonable range)
        df_filtered = self.listings[self.listings['price'] < 100000]
        sns.boxplot(data=df_filtered, x='room_type', y='price', ax=axes[0])
        axes[0].set_title('Price by Room Type (Boxplot)')
        axes[0].set_xlabel('Room Type')
        axes[0].set_ylabel('Price (JPY)')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Bar chart (mean)
        mean_prices = self.listings.groupby('room_type')['price'].mean().sort_values()
        mean_prices.plot(kind='bar', ax=axes[1])
        axes[1].set_title('Average Price by Room Type')
        axes[1].set_ylabel('Average Price (JPY)')
        axes[1].set_xlabel('')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / 'price_by_room_type.png'
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"💾 Saved to {path}")
            return str(path)
        
        plt.show()
        return "displayed"

    def plot_price_by_neighborhood(self, top_n: int = 10, order: str = "expensive", save: bool = True) -> str:
        """Plot average price by neighbourhood. order='expensive' or 'cheapest'."""
        if 'neighbourhood_cleansed' not in self.listings.columns:
            return "Neighborhood data not available"

        cheapest = (order == "cheapest")
        nb_prices = self.listings.groupby('neighbourhood_cleansed')['price'].agg(['mean', 'count'])
        nb_prices = nb_prices.sort_values('mean', ascending=cheapest).head(top_n)

        title = f'Top {top_n} {"Cheapest" if cheapest else "Most Expensive"} Neighborhoods'
        color = '#5BC8AF' if cheapest else 'skyblue'

        fig, ax = plt.subplots(figsize=(12, 6))
        nb_prices['mean'].plot(kind='bar', ax=ax, color=color)
        ax.set_title(title)
        ax.set_ylabel('Average Price (JPY)')
        ax.set_xlabel('Neighborhood')
        ax.tick_params(axis='x', rotation=45)

        for i, (idx, row) in enumerate(nb_prices.iterrows()):
            ax.text(i, row['mean'] + 200, f"n={int(row['count'])}", ha='center', fontsize=8)

        plt.tight_layout()

        if save:
            filename = 'price_by_neighborhood_cheapest.png' if cheapest else 'price_by_neighborhood.png'
            path = self.output_dir / filename
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"💾 Saved to {path}")
            return str(path)

        plt.show()
        return "displayed"

    def plot_correlation_heatmap(self, save: bool = True) -> str:
        """Plot correlation heatmap of numeric features."""
        # Select numeric columns
        numeric_cols = ['price', 'accommodates', 'bathrooms', 'bedrooms', 'beds',
                       'number_of_reviews', 'review_scores_rating', 'availability_365']
        numeric_cols = [col for col in numeric_cols if col in self.listings.columns]
        
        # Correlation matrix
        corr = self.listings[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        ax.set_title('Correlation Heatmap')
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / 'correlation_heatmap.png'
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"💾 Saved to {path}")
            return str(path)
        
        plt.show()
        return "displayed"

    def create_summary_report(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        report = {
            'total_listings': len(self.listings),
            'total_listings_with_price': self.listings['price'].count(),
            'avg_price': self.listings['price'].mean(),
            'median_price': self.listings['price'].median(),
            'min_price': self.listings['price'].min(),
            'max_price': self.listings['price'].max(),
            'avg_review_score': self.listings['review_scores_rating'].mean() if 'review_scores_rating' in self.listings.columns else None,
            'total_neighborhoods': self.listings['neighbourhood_cleansed'].nunique() if 'neighbourhood_cleansed' in self.listings.columns else 0,
        }
        
        # Add room type breakdown
        if 'room_type' in self.listings.columns:
            report['room_type_counts'] = self.listings['room_type'].value_counts().to_dict()
            report['room_type_avg_price'] = self.listings.groupby('room_type')['price'].mean().round(0).to_dict()
        
        # Add top neighborhoods
        if 'neighbourhood_cleansed' in self.listings.columns:
            report['top_5_expensive'] = self.listings.groupby('neighbourhood_cleansed')['price']                 .mean().sort_values(ascending=False).head(5).round(0).to_dict()
            report['top_5_cheapest'] = self.listings.groupby('neighbourhood_cleansed')['price']                 .mean().sort_values().head(5).round(0).to_dict()
        
        return report

    def generate_all_plots(self) -> list:
        """Generate all plots and return their paths."""
        paths = []
        paths.append(self.plot_price_distribution())
        paths.append(self.plot_price_by_room_type())
        paths.append(self.plot_price_by_neighborhood())
        paths.append(self.plot_correlation_heatmap())
        return paths

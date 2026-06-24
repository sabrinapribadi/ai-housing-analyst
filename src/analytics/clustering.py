"""
Geospatial clustering module for Airbnb Tokyo data.
Identifies market segments using K-Means on coordinates.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import folium
from typing import Optional, Dict, Any, Tuple
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeospatialClusterer:
    """Cluster Airbnb listings by location and price."""

    def __init__(self, listings_df: pd.DataFrame):
        """
        Initialize with cleaned listings data.

        Args:
            listings_df: Cleaned listings dataframe with coordinates
        """
        self.listings = listings_df
        self.model = None
        self.scaler = StandardScaler()
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)

    def prepare_features(self, include_price: bool = True) -> np.ndarray:
        """
        Prepare features for clustering.

        Args:
            include_price: Whether to include price in clustering features

        Returns:
            Normalized feature matrix
        """
        features = ['latitude', 'longitude']
        if include_price and 'price' in self.listings.columns:
            features.append('price')

        X = self.listings[features].dropna().values
        X_scaled = self.scaler.fit_transform(X)

        logger.info(f"✅ Prepared {len(X_scaled):,} rows with features: {features}")
        return X_scaled

    def find_optimal_clusters(self, max_k: int = 10) -> Dict[str, Any]:
        """
        Find optimal number of clusters using elbow method.

        Args:
            max_k: Maximum number of clusters to test

        Returns:
            Dictionary with inertia values and optimal k
        """
        X = self.prepare_features()

        inertias = []
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)

        diffs = np.diff(inertias)
        elbow = np.argmin(diffs[1:]) + 2

        logger.info(f"📊 Optimal clusters: {elbow} (based on elbow method)")

        return {
            'k_values': list(range(1, max_k + 1)),
            'inertias': inertias,
            'optimal_k': elbow
        }

    def cluster(self, n_clusters: int = 5, include_price: bool = True) -> pd.DataFrame:
        """
        Cluster listings by location (and optionally price).

        Args:
            n_clusters: Number of clusters to create
            include_price: Whether to include price in clustering

        Returns:
            DataFrame with cluster labels
        """
        X = self.prepare_features(include_price)

        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.listings['cluster'] = pd.NA

        valid_mask = self.listings[['latitude', 'longitude']].notna().all(axis=1)
        if include_price:
            valid_mask = valid_mask & self.listings['price'].notna()

        self.listings.loc[valid_mask, 'cluster'] = self.model.fit_predict(X)

        cluster_names = self._generate_cluster_names()
        self.listings['cluster_name'] = self.listings['cluster'].map(cluster_names)

        logger.info(f"✅ Assigned {len(self.listings[self.listings['cluster'].notna()]):,} listings to {n_clusters} clusters")

        self._log_cluster_stats()

        return self.listings

    def _generate_cluster_names(self) -> Dict[int, str]:
        """Generate meaningful cluster names based on characteristics."""
        if self.model is None:
            return {}

        cluster_stats = self.listings[self.listings['cluster'].notna()].groupby('cluster').agg({
            'price': 'mean',
            'latitude': 'mean',
            'longitude': 'mean'
        })

        names = {}
        for cluster_id, row in cluster_stats.iterrows():
            price_level = 'Luxury' if row['price'] > 30000 else 'Premium' if row['price'] > 20000 else 'Standard' if row['price'] > 12000 else 'Budget'

            lat = row['latitude']
            if lat > 35.75:
                area = 'North'
            elif lat > 35.65:
                area = 'Central'
            else:
                area = 'South'

            names[cluster_id] = f"{price_level} {area}"

        return names

    def _log_cluster_stats(self):
        """Log statistics for each cluster."""
        stats = self.listings[self.listings['cluster'].notna()].groupby('cluster').agg({
            'price': ['count', 'mean', 'std'],
            'latitude': 'mean',
            'longitude': 'mean'
        })

        logger.info("📊 Cluster Statistics:")
        for cluster_id, row in stats.iterrows():
            name = self.listings[self.listings['cluster'] == cluster_id]['cluster_name'].iloc[0]
            count = row[('price', 'count')]
            price_mean = row[('price', 'mean')]
            logger.info(f"  Cluster {cluster_id} ({name}): {count} listings, Avg Price: ¥{price_mean:,.0f}")

    def plot_clusters(self, save: bool = True) -> str:
        """Plot clusters on a 2D scatter plot."""
        df = self.listings[self.listings['cluster'].notna()]

        fig, ax = plt.subplots(figsize=(12, 10))

        scatter = ax.scatter(
            df['longitude'],
            df['latitude'],
            c=df['cluster'],
            cmap='tab10',
            s=20,
            alpha=0.6
        )

        ax.set_title('Geospatial Clusters of Airbnb Listings in Tokyo')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        legend_labels = df.groupby('cluster')['cluster_name'].first().to_dict()
        handles = scatter.legend_elements()[0]
        ax.legend(handles, [legend_labels[i] for i in range(len(handles))], title='Cluster')

        plt.tight_layout()

        if save:
            path = self.output_dir / 'geospatial_clusters.png'
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"💾 Saved to {path}")
            return str(path)

        plt.show()
        return "displayed"

    def create_interactive_map(self) -> folium.Map:
        """Create an interactive folium map with cluster markers."""
        df = self.listings[self.listings['cluster'].notna()]

        map_center = [df['latitude'].mean(), df['longitude'].mean()]
        m = folium.Map(location=map_center, zoom_start=11)

        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige']

        sample_df = df.sample(n=min(500, len(df)), random_state=42)

        for _, row in sample_df.iterrows():
            cluster_id = int(row['cluster'])
            color = colors[cluster_id % len(colors)]

            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                popup=f"Price: ¥{row['price']:.0f}<br>Cluster: {row['cluster_name']}<br>Neighborhood: {row.get('neighbourhood_cleansed', 'Unknown')}",
                color=color,
                fill=True,
                fill_opacity=0.6
            ).add_to(m)

        logger.info("🗺️ Interactive map created")
        return m

    def get_cluster_summary(self) -> pd.DataFrame:
        """Get summary of clusters."""
        df = self.listings[self.listings['cluster'].notna()]

        summary = df.groupby(['cluster', 'cluster_name']).agg({
            'id': 'count',
            'price': ['mean', 'min', 'max'],
            'latitude': 'mean',
            'longitude': 'mean'
        }).round(2)

        summary.columns = ['count', 'avg_price', 'min_price', 'max_price', 'avg_lat', 'avg_lon']
        return summary

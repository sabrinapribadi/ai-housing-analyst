import sys
sys.path.append('.')
import pandas as pd
from src.analytics.clustering import GeospatialClusterer

# Load data
print("Loading data...")
df = pd.read_parquet('data/processed/listings_processed.parquet')
print(f"✅ Loaded {len(df):,} listings")

# Initialize clusterer
clusterer = GeospatialClusterer(df)

# Find optimal clusters
print('\n' + '=' * 60)
print('🔍 Finding Optimal Clusters...')
print('=' * 60)
result = clusterer.find_optimal_clusters(max_k=8)
print(f"Optimal number of clusters: {result['optimal_k']}")

# Cluster the data
print('\n' + '=' * 60)
print('🏠 Clustering Listings...')
print('=' * 60)
df_clustered = clusterer.cluster(n_clusters=5, include_price=True)
print(f"✅ Clustered {len(df_clustered[df_clustered['cluster'].notna()]):,} listings")

# Generate plot
print('\n' + '=' * 60)
print('📈 Generating Cluster Plot...')
print('=' * 60)
plot_path = clusterer.plot_clusters()
print(f'  ✅ {plot_path}')

# Create interactive map
print('\n' + '=' * 60)
print('🗺️ Creating Interactive Map...')
print('=' * 60)
map_obj = clusterer.create_interactive_map()
map_obj.save('outputs/cluster_map.html')
print('  ✅ outputs/cluster_map.html')

# Show summary
print('\n' + '=' * 60)
print('📊 Cluster Summary')
print('=' * 60)
summary = clusterer.get_cluster_summary()
print(summary)

print('\n✅ Done! Check outputs/ for cluster plots and map.')

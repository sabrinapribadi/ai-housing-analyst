import sys
sys.path.append('.')

from src.data.loader import DataLoader
from src.data.cleaner import DataCleaner
from src.data.feature_engineer import FeatureEngineer
import pandas as pd

print("Loading data...")
loader = DataLoader()
raw_data = loader.load_all(sample_listings=None, sample_calendar=50000, sample_reviews=20000)

print("Cleaning...")
cleaner = DataCleaner()
cleaned_data = cleaner.clean_all(raw_data)

print("Engineering features...")
engineer = FeatureEngineer()
final_data = engineer.engineer_all(cleaned_data)

df = final_data['listings']
print(f"\nDone. {len(df):,} listings loaded.")
print(f"Columns: {len(df.columns)}")
print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
print(f"\nPrice stats:")
print(df['price'].describe())

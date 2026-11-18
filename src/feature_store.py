"""Build and reuse an offline CSV feature store."""
from pathlib import Path
import pandas as pd
from .preprocessing import load_and_engineer, ROOT

def build_feature_store(input_file=None, output_file=None):
    input_file = Path(input_file or ROOT/"data"/"pond_nights_v2_rainy.csv")
    output_file = Path(output_file or ROOT/"feature_store"/"pond_night_features.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    features = load_and_engineer(input_file)
    features.to_csv(output_file, index=False)
    print(f"Saved {len(features)} engineered rows to {output_file}")
    return output_file

def load_features(path=None):
    path = Path(path or ROOT/"feature_store"/"pond_night_features.csv")
    return pd.read_csv(path)

if __name__ == "__main__":
    build_feature_store()

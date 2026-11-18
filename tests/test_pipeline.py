from pathlib import Path
import pandas as pd
from src.preprocessing import load_and_engineer, ROOT
from src.feature_store import build_feature_store, load_features

def test_dataset_schema_and_labels():
    df = load_and_engineer(ROOT/"data"/"pond_nights_v1.csv")
    assert len(df) >= 250
    assert set(df["oxygen_risk"].unique()).issubset({"High","Normal"})

def test_engineered_features_are_created():
    df = load_and_engineer(ROOT/"data"/"pond_nights_v1.csv")
    assert "feed_per_unit_biomass" in df.columns
    assert "aeration_hours_per_area" in df.columns
    assert df[["feed_per_unit_biomass","aeration_hours_per_area"]].notna().all().all()

def test_feature_store_round_trip(tmp_path):
    output = tmp_path/"features.csv"
    build_feature_store(ROOT/"data"/"pond_nights_v2_rainy.csv", output)
    loaded = pd.read_csv(output)
    assert len(loaded) == 300
    assert "feed_per_unit_biomass" in loaded.columns

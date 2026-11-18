"""Reusable preprocessing and feature engineering."""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
TARGET = "oxygen_risk"
NUMERIC = ["pond_area_acres","stocking_density_per_m2","do_8pm_mg_l","water_temp_c",
           "daily_feed_kg","planned_aerator_hours","expected_rainfall_mm",
           "feed_per_unit_biomass","aeration_hours_per_area"]

def engineer_features(df):
    df = df.copy()
    biomass = (df["pond_area_acres"] * 4046.86 * df["stocking_density_per_m2"]) / 1000
    df["feed_per_unit_biomass"] = df["daily_feed_kg"] / biomass.clip(lower=0.01)
    df["aeration_hours_per_area"] = df["planned_aerator_hours"] / df["pond_area_acres"].clip(lower=0.01)
    return df

def build_preprocessor():
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    return ColumnTransformer([("numeric", numeric_pipe, NUMERIC)], remainder="drop")

def load_and_engineer(csv_path):
    df = pd.read_csv(csv_path)
    required = set(["pond_area_acres","stocking_density_per_m2","do_8pm_mg_l","water_temp_c",
                    "daily_feed_kg","planned_aerator_hours","expected_rainfall_mm",TARGET])
    missing = required - set(df.columns)
    if missing: raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty: raise ValueError("Dataset is empty")
    if not set(df[TARGET].dropna().unique()).issubset({"High","Normal"}):
        raise ValueError("oxygen_risk must contain High/Normal labels")
    return engineer_features(df)

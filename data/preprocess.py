"""
Centralized Data Preprocessing
--------------------------------
Loads raw IoT telemetry CSV,
creates 3-class risk labels,
normalises features,
saves processed CSV and scaler.

Label logic:
  2 = Critical : smoke > 0.10  OR  co > 0.005
  1 = Warning  : temp  > 90   OR  lpg > 0.007
  0 = Normal
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib


# ======================================================
# CONFIG LOADER (simple version)
# ======================================================

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

RAW_PATH      = Path(config["data"]["raw_path"])
PROCESSED_DIR = Path(config["data"]["processed_path"])

FEATURE_COLS = ["co", "humidity", "light", "lpg", "motion", "smoke", "temp"]
LABEL_COL    = "label"


# ======================================================
# LABEL CREATION
# ======================================================

def create_labels(df: pd.DataFrame) -> pd.Series:
    conditions = [
        (df["smoke"] > 0.10) | (df["co"] > 0.005),
        (df["temp"]  > 90.0) | (df["lpg"] > 0.007),
    ]
    choices = [2, 1]
    return np.select(conditions, choices, default=0).astype(int)


# ======================================================
# PREPROCESS FUNCTION
# ======================================================

def preprocess():
    print(f"Loading raw data from {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)
    print("Raw shape:", df.shape)

    # Convert timestamp
    df["ts"] = pd.to_datetime(df["ts"], unit="s")

    df = df.sort_values("ts").reset_index(drop=True)

    # Ensure integer types
    df["light"]  = df["light"].astype(int)
    df["motion"] = df["motion"].astype(int)

    # Create labels
    df[LABEL_COL] = create_labels(df)

    print("Label distribution:")
    print(df[LABEL_COL].value_counts())

    # Normalize features
    scaler = StandardScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Save scaler
    scaler_path = PROCESSED_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print("Scaler saved to:", scaler_path)

    # Save processed CSV
    out_path = PROCESSED_DIR / "processed.csv"
    df.to_csv(out_path, index=False)

    print("Processed data saved to:", out_path)
    print("Processed shape:", df.shape)

    return df


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":
    preprocess()

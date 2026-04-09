import pandas as pd
import glob, os, json
from datetime import datetime

def load_raw(path="data/raw/*.json") -> pd.DataFrame:
    files = glob.glob(path)
    if not files:
        raise FileNotFoundError("No raw JSON files found. Run ingestion.py first.")
    records = []
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
            if isinstance(data, list):
                records.extend(data)
    return pd.DataFrame(records)

def clean(df: pd.DataFrame) -> pd.DataFrame:
    numeric = ["aqi","pm25","pm10","no2","o3","co","so2",
                "temperature","humidity","wind_speed","pressure"]
    df = df.dropna(subset=["city","aqi","timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.drop_duplicates(subset=["city","timestamp"])
    return df

def transform(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"]       = df["timestamp"].dt.month
    df["aqi_category"] = pd.cut(
        df["aqi"],
        bins=[0, 50, 100, 150, 200, 999],
        labels=["Good","Moderate","Unhealthy for Sensitive Groups","Unhealthy","Very Unhealthy"]
    )
    return df

def run_pipeline():
    print("Loading raw data...")
    raw = load_raw()
    print(f"  Rows loaded: {len(raw)}")

    print("Cleaning...")
    clean_df = clean(raw)

    print("Transforming...")
    final = transform(clean_df)
    print(final[["city","aqi","aqi_category","temperature","hour"]].to_string())

    out = "data/processed"
    os.makedirs(out, exist_ok=True)
    final.to_parquet(f"{out}/processed.parquet", index=False)
    print(f"\nSaved to {out}/processed.parquet")

if __name__ == "__main__":
    run_pipeline()
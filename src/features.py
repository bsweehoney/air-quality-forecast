import pandas as pd
import numpy as np
import os

def load_processed(path="data/processed") -> pd.DataFrame:
    return pd.read_parquet(path)

def add_lag_features(df: pd.DataFrame, target="aqi", lags=[1, 2, 3, 6, 12, 24]) -> pd.DataFrame:
    df = df.sort_values(["city", "timestamp"])
    for lag in lags:
        df[f"{target}_lag_{lag}"] = df.groupby("city")[target].shift(lag)
    return df

def add_rolling_features(df: pd.DataFrame, target="aqi", windows=[3, 6, 12, 24]) -> pd.DataFrame:
    for w in windows:
        df[f"{target}_roll_mean_{w}"] = (df.groupby("city")[target]
                                           .transform(lambda x: x.rolling(w, min_periods=1).mean()))
        df[f"{target}_roll_std_{w}"]  = (df.groupby("city")[target]
                                           .transform(lambda x: x.rolling(w, min_periods=1).std()))
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df["temp_humidity_interaction"] = df["temperature"] * df["humidity"]
    df["wind_aqi_ratio"]            = df["aqi"] / (df["wind_speed"] + 1)
    df = df.dropna(subset=["aqi_lag_24"])   # need at least 24h of history
    return df

if __name__ == "__main__":
    df = load_processed()
    df = build_features(df)
    os.makedirs("data/features", exist_ok=True)
    df.to_parquet("data/features/features.parquet", index=False)
    print(f"Feature set: {df.shape[0]} rows × {df.shape[1]} columns")
    print(df[["city","aqi","aqi_lag_1","aqi_roll_mean_6"]].head())
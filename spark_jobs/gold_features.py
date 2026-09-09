import boto3, json, os, io
from botocore.config import Config
from datetime import datetime
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

LOCALSTACK = os.getenv("LOCALSTACK_ENDPOINT", "http://localstack:4566")

def get_s3():
    return boto3.client(
        "s3",
        endpoint_url=LOCALSTACK,
        aws_access_key_id="test",
        aws_secret_access_key="test",
        region_name="us-east-1",
        config=Config(signature_version="s3v4")
    )

def read_silver(s3) -> pd.DataFrame:
    paginator = s3.get_paginator("list_objects_v2")
    pages     = paginator.paginate(Bucket="silver-air-quality")
    frames    = []
    for page in pages:
        for obj in page.get("Contents", []):
            body = s3.get_object(
                Bucket="silver-air-quality",
                Key=obj["Key"]
            )["Body"].read()
            frames.append(pq.read_table(io.BytesIO(body)).to_pandas())
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    print(f"Read {len(df)} rows from Silver")
    return df

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["city","timestamp"])
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"aqi_lag_{lag}"] = df.groupby("city")["aqi"].shift(lag)
    return df

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    for window in [3, 6, 12, 24]:
        df[f"aqi_roll_mean_{window}"] = (
            df.groupby("city")["aqi"]
              .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        df[f"aqi_roll_std_{window}"] = (
            df.groupby("city")["aqi"]
              .transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
        )
    return df

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    if "temperature" in df.columns and "humidity" in df.columns:
        df["temp_humidity"] = df["temperature"] * df["humidity"]
    if "wind_speed" in df.columns:
        df["wind_aqi_ratio"] = df["aqi"] / (df["wind_speed"] + 1)
    return df

def validate_gold(df: pd.DataFrame):
    assert "aqi_lag_1"       in df.columns, "Missing lag feature!"
    assert "aqi_roll_mean_6" in df.columns, "Missing rolling feature!"
    assert df["aqi"].notna().all(),          "AQI nulls in Gold!"
    print(f"✓ Gold validation passed — {len(df)} rows, {len(df.columns)} columns")

def write_gold(s3, df: pd.DataFrame):
    table  = pa.Table.from_pandas(df)
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    buffer.seek(0)
    key = f"features/gold_features.parquet"
    s3.put_object(
        Bucket="gold-air-quality",
        Key=key,
        Body=buffer.getvalue(),
        ContentType="application/octet-stream"
    )
    print(f"✓ Written to s3://gold-air-quality/{key}")

def run():
    s3 = get_s3()
    df = read_silver(s3)

    if df.empty:
        print("No Silver data found — skipping")
        return

    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_interaction_features(df)
    validate_gold(df)
    write_gold(s3, df)
    print(f"\nGold layer complete — {len(df)} rows, {len(df.columns)} features")

if __name__ == "__main__":
    run()
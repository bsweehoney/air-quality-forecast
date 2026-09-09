import boto3, json, os
from botocore.config import Config
from datetime import datetime
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import io

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

def read_bronze(s3, date_prefix) -> pd.DataFrame:
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket="bronze-air-quality", Prefix=date_prefix)
    records = []
    for page in pages:
        for obj in page.get("Contents", []):
            body = s3.get_object(Bucket="bronze-air-quality", Key=obj["Key"])["Body"].read()
            records.append(json.loads(body))
    print(f"Read {len(records)} records from Bronze")
    return pd.DataFrame(records)

def clean(df: pd.DataFrame) -> pd.DataFrame:
    numeric = ["aqi","pm25","pm10","no2","o3","co","so2"]
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
        labels=["Good","Moderate","Unhealthy-Sensitive","Unhealthy","Very-Unhealthy"]
    ).astype(str)
    return df

def validate(df: pd.DataFrame):
    assert df["aqi"].notna().all(),       "AQI has nulls!"
    assert df["city"].notna().all(),      "City has nulls!"
    assert (df["aqi"] >= 0).all(),        "AQI has negative values!"
    assert (df["aqi"] <= 500).all(),      "AQI exceeds 500!"
    assert df["timestamp"].notna().all(), "Timestamp has nulls!"
    print(f"✓ Validation passed — {len(df)} rows, {len(df.columns)} columns")

def write_silver(s3, df: pd.DataFrame, date_prefix: str):
    table  = pa.Table.from_pandas(df)
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    buffer.seek(0)
    key = f"{date_prefix}silver.parquet"
    s3.put_object(
        Bucket="silver-air-quality",
        Key=key,
        Body=buffer.getvalue(),
        ContentType="application/octet-stream"
    )
    print(f"✓ Written to s3://silver-air-quality/{key}")

def run(date_str=None):
    if not date_str:
        date_str = datetime.utcnow().strftime("%Y/%m/%d")
    y, m, d   = date_str.split("/")
    prefix    = f"year={y}/month={m}/day={d}/"

    s3  = get_s3()
    df  = read_bronze(s3, prefix)

    if df.empty:
        print("No data in Bronze for this date — skipping")
        return

    df  = clean(df)
    df  = transform(df)
    validate(df)
    write_silver(s3, df, prefix)
    print(f"\nSilver layer complete — {len(df)} rows processed")

if __name__ == "__main__":
    run()
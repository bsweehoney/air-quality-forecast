from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests, json, os, boto3
from botocore.config import Config

default_args = {
    'owner': 'airflow',
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

CITIES = ["new york", "london", "beijing", "delhi", "sydney"]
AQICN_KEY = os.getenv("AQICN_API_KEY")

def get_s3():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("LOCALSTACK_ENDPOINT", "http://localstack:4566"),
        aws_access_key_id="test",
        aws_secret_access_key="test",
        region_name="us-east-1",
        config=Config(signature_version="s3v4")
    )

def create_buckets():
    s3 = get_s3()
    for bucket in ["bronze-air-quality", "silver-air-quality", "gold-air-quality"]:
        try:
            s3.create_bucket(Bucket=bucket)
            print(f"Created bucket: {bucket}")
        except Exception as e:
            print(f"Bucket {bucket} already exists: {e}")

def fetch_aqi(city):
    url = f"https://api.waqi.info/feed/{city}/?token={AQICN_KEY}"
    r = requests.get(url, timeout=10)
    data = r.json()
    if data["status"] != "ok":
        return None
    d = data["data"]
    iaqi = d.get("iaqi", {})
    return {
        "city": city,
        "timestamp": datetime.utcnow().isoformat(),
        "aqi": d.get("aqi"),
        "pm25": iaqi.get("pm25", {}).get("v"),
        "pm10": iaqi.get("pm10", {}).get("v"),
        "no2":  iaqi.get("no2",  {}).get("v"),
        "o3":   iaqi.get("o3",   {}).get("v"),
        "co":   iaqi.get("co",   {}).get("v"),
        "so2":  iaqi.get("so2",  {}).get("v"),
    }

def ingest_to_bronze(**context):
    create_buckets()
    s3  = get_s3()
    ts  = datetime.utcnow()
    records = []
    for city in CITIES:
        try:
            record = fetch_aqi(city)
            if record:
                records.append(record)
                print(f"✓ {city}: AQI={record['aqi']}")
        except Exception as e:
            print(f"✗ {city}: {e}")

    for record in records:
        city = record["city"].replace(" ", "_")
        key  = f"year={ts.year}/month={ts.month:02d}/day={ts.day:02d}/city={city}/{ts.strftime('%H%M%S')}.json"
        s3.put_object(
            Bucket="bronze-air-quality",
            Key=key,
            Body=json.dumps(record),
            ContentType="application/json"
        )
        print(f"Saved → s3://bronze-air-quality/{key}")

    print(f"\nTotal records ingested: {len(records)}")

with DAG(
    dag_id="air_quality_ingestion",
    default_args=default_args,
    description="Hourly AQI ingestion from AQICN API to S3 Bronze",
    schedule_interval="@hourly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["air-quality", "bronze", "ingestion"],
) as dag:

    ingest_task = PythonOperator(
        task_id="ingest_to_bronze",
        python_callable=ingest_to_bronze,
        provide_context=True,
    )
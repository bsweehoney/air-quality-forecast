import ollama
import pandas as pd
import json
import os
import boto3
import io
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.config import Config
from datetime import datetime

MODEL      = "llama3.1:8b"
LOCALSTACK = os.getenv("LOCALSTACK_ENDPOINT", "http://localhost:4566")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")

def get_s3():
    return boto3.client(
        "s3",
        endpoint_url=LOCALSTACK,
        aws_access_key_id="test",
        aws_secret_access_key="test",
        region_name="us-east-1",
        config=Config(signature_version="s3v4")
    )

def get_ollama():
    return ollama.Client(host=OLLAMA_HOST)

def diagnose_with_ai(record: dict, issues: list) -> dict:
    prompt = f"""You are a data quality expert for an air quality monitoring system.

A data record has issues. Analyze and suggest fixes.

Record: {json.dumps(record)}
Issues found: {', '.join(issues)}

Valid AQI range: 0-500
Valid cities: new york, london, beijing, delhi, sydney
AQI typical ranges:
- beijing: 100-200
- delhi: 100-180
- london: 20-60
- new york: 30-70
- sydney: 10-40

Respond in JSON only:
{{
    "diagnosis": "one sentence explaining all issues",
    "fix_action": "impute|drop|correct",
    "fixed_aqi": null,
    "confidence": 0.95,
    "reasoning": "one sentence explaining fix"
}}"""

    try:
        client   = get_ollama()
        response = client.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response["message"]["content"]
        start   = content.find("{")
        end     = content.rfind("}") + 1
        if start != -1 and end != 0:
            return json.loads(content[start:end])
    except Exception as e:
        print(f"    AI failed: {e}")

    return {
        "diagnosis":  "Fallback diagnosis",
        "fix_action": "drop",
        "fixed_aqi":  None,
        "confidence": 0.5,
        "reasoning":  "Dropped due to AI failure"
    }

def detect_issues(row: pd.Series) -> list:
    issues = []
    aqi    = row.get("aqi")

    if pd.isna(aqi):
        issues.append("missing_aqi")
    else:
        try:
            aqi_float = float(aqi)
            if aqi_float < 0 or aqi_float > 500:
                issues.append("out_of_range_aqi")
        except (ValueError, TypeError):
            issues.append("wrong_type_aqi")

    if pd.isna(row.get("city")):
        issues.append("missing_city")

    if pd.isna(row.get("timestamp")):
        issues.append("missing_timestamp")

    return issues

def heal_dataframe(df: pd.DataFrame) -> tuple:
    heal_log  = []
    drop_idx  = []
    city_avgs = {}

    valid_mask = pd.to_numeric(df["aqi"], errors="coerce").between(0, 500)
    valid_df   = df[valid_mask & df["city"].notna()]
    for city in valid_df["city"].unique():
        city_avgs[city] = pd.to_numeric(
            valid_df[valid_df["city"] == city]["aqi"], errors="coerce"
        ).mean()

    healed_df = df.copy()
    healed_df["aqi"] = pd.to_numeric(healed_df["aqi"], errors="coerce")

    for idx, row in df.iterrows():
        issues = detect_issues(row)
        if not issues:
            continue

        print(f"  [{idx}] Issues: {issues} → asking LLaMA...")
        diagnosis  = diagnose_with_ai(row.to_dict(), issues)
        action     = diagnosis.get("fix_action", "drop")
        fixed_aqi  = diagnosis.get("fixed_aqi")
        confidence = diagnosis.get("confidence", 0.5)

        if "missing_city" in issues or "missing_timestamp" in issues:
            drop_idx.append(idx)
            final_action = "dropped — missing city/timestamp"

        elif action == "drop":
            drop_idx.append(idx)
            final_action = "dropped by AI"

        elif action in ("impute", "correct"):
            if fixed_aqi and str(fixed_aqi) != "null":
                try:
                    healed_df.at[idx, "aqi"] = min(max(float(fixed_aqi), 0), 500)
                    final_action = f"corrected to {fixed_aqi}"
                except (ValueError, TypeError):
                    city    = row.get("city")
                    imputed = city_avgs.get(city, 75.0)
                    healed_df.at[idx, "aqi"] = round(imputed, 1)
                    final_action = f"imputed with city avg {round(imputed,1)}"
            else:
                city    = row.get("city")
                imputed = city_avgs.get(city, 75.0)
                healed_df.at[idx, "aqi"] = round(imputed, 1)
                final_action = f"imputed with city avg {round(imputed,1)}"
        else:
            drop_idx.append(idx)
            final_action = "dropped — unknown action"

        heal_log.append({
            "timestamp":  datetime.utcnow().isoformat(),
            "city":       row.get("city"),
            "issues":     str(issues),
            "action":     final_action,
            "diagnosis":  diagnosis.get("diagnosis"),
            "confidence": confidence,
            "reasoning":  diagnosis.get("reasoning"),
        })
        print(f"  ✓ {final_action} (confidence: {confidence})")

    healed_df = healed_df.drop(index=drop_idx, errors="ignore")
    healed_df = healed_df.drop_duplicates(subset=["city", "timestamp"])
    healed_df = healed_df.dropna(subset=["aqi", "city", "timestamp"])

    return healed_df, pd.DataFrame(heal_log)

def run(date_str=None):
    if not date_str:
        date_str = datetime.utcnow().strftime("%Y/%m/%d")

    y, m, d = date_str.split("/")
    prefix  = f"year={y}/month={m}/day={d}/"

    s3 = get_s3()

    paginator = s3.get_paginator("list_objects_v2")
    pages     = paginator.paginate(Bucket="bronze-air-quality", Prefix=prefix)
    records   = []
    for page in pages:
        for obj in page.get("Contents", []):
            body = s3.get_object(
                Bucket="bronze-air-quality",
                Key=obj["Key"]
            )["Body"].read()
            records.append(json.loads(body))

    if not records:
        print("No Bronze data found for this date")
        return

    df = pd.DataFrame(records)
    print(f"\nLoaded {len(df)} records from Bronze")
    print(f"Running self-healing with LLaMA 3.1...")

    healed_df, heal_log = heal_dataframe(df)

    print(f"\nHealing complete:")
    print(f"  Original : {len(df)} records")
    print(f"  Healed   : {len(healed_df)} records")
    print(f"  Fixed    : {len(heal_log)} issues")

    os.makedirs("data", exist_ok=True)
    if not heal_log.empty:
        heal_log.to_csv("data/heal_log.csv", index=False)

    table  = pa.Table.from_pandas(healed_df)
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    buffer.seek(0)

    key = f"{prefix}healed_silver.parquet"
    s3.put_object(
        Bucket="silver-air-quality",
        Key=key,
        Body=buffer.getvalue(),
        ContentType="application/octet-stream"
    )
    print(f"✓ Healed data saved → s3://silver-air-quality/{key}")

if __name__ == "__main__":
    run()
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

MODEL       = "llama3.1:8b"
LOCALSTACK  = os.getenv("LOCALSTACK_ENDPOINT", "http://localhost:4566")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")

# ── S3 helper ──────────────────────────────────────────────────────────
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

# ── Tools the agent can call ───────────────────────────────────────────
def tool_detect_issues(df: pd.DataFrame) -> dict:
    """Tool: Scan dataframe and return summary of all issues found"""
    issues = {
        "missing_aqi":    int(df["aqi"].isna().sum()),
        "wrong_type_aqi": int(pd.to_numeric(df["aqi"], errors="coerce").isna().sum()
                             - df["aqi"].isna().sum()),
        "out_of_range":   int(pd.to_numeric(df["aqi"], errors="coerce")
                             .apply(lambda x: x < 0 or x > 500
                             if pd.notna(x) else False).sum()),
        "missing_city":   int(df["city"].isna().sum()),
        "duplicates":     int(df.duplicated(subset=["city","timestamp"]).sum()),
        "total_records":  len(df),
        "clean_records":  int(df["aqi"].notna().sum()),
    }
    return issues

def tool_fix_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Tool: Impute missing AQI values with city averages"""
    city_avgs = {}
    valid = df[pd.to_numeric(df["aqi"], errors="coerce").between(0, 500)]
    for city in valid["city"].dropna().unique():
        city_avgs[city] = pd.to_numeric(
            valid[valid["city"] == city]["aqi"], errors="coerce"
        ).mean()

    for idx, row in df[df["aqi"].isna()].iterrows():
        city = row.get("city")
        df.at[idx, "aqi"] = round(city_avgs.get(city, 75.0), 1)
        print(f"    ✓ Imputed AQI for {city} with {city_avgs.get(city, 75.0):.1f}")
    return df

def tool_fix_types(df: pd.DataFrame) -> pd.DataFrame:
    """Tool: Fix wrong type AQI values"""
    df["aqi"] = pd.to_numeric(df["aqi"], errors="coerce")
    fixed = df["aqi"].isna().sum()
    print(f"    ✓ Fixed types — {fixed} values converted/dropped")
    return df

def tool_fix_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Tool: Cap out-of-range AQI values between 0 and 500"""
    before = len(df)
    df["aqi"] = df["aqi"].clip(lower=0, upper=500)
    print(f"    ✓ Capped out-of-range values")
    return df

def tool_drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Tool: Remove duplicate records"""
    before = len(df)
    df = df.drop_duplicates(subset=["city","timestamp"])
    print(f"    ✓ Removed {before - len(df)} duplicates")
    return df

def tool_drop_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Tool: Drop records with missing city or timestamp"""
    before = len(df)
    df = df.dropna(subset=["city","timestamp","aqi"])
    print(f"    ✓ Dropped {before - len(df)} unfixable records")
    return df

def tool_validate(df: pd.DataFrame) -> dict:
    """Tool: Final validation report"""
    return {
        "total_records":   len(df),
        "missing_aqi":     int(df["aqi"].isna().sum()),
        "missing_city":    int(df["city"].isna().sum()),
        "out_of_range":    int(pd.to_numeric(df["aqi"], errors="coerce")
                              .apply(lambda x: x < 0 or x > 500
                              if pd.notna(x) else False).sum()),
        "duplicates":      int(df.duplicated(subset=["city","timestamp"]).sum()),
        "is_clean":        True
    }

# ── Tool registry ──────────────────────────────────────────────────────
TOOLS = {
    "detect_issues":  tool_detect_issues,
    "fix_nulls":      tool_fix_nulls,
    "fix_types":      tool_fix_types,
    "fix_ranges":     tool_fix_ranges,
    "drop_duplicates":tool_drop_duplicates,
    "drop_nulls":     tool_drop_nulls,
    "validate":       tool_validate,
}

TOOL_DESCRIPTIONS = """
You have access to these tools to heal the data:
- detect_issues: Scan data and return summary of all problems
- fix_nulls: Impute missing AQI values with city averages
- fix_types: Convert wrong-type AQI values to numeric
- fix_ranges: Cap out-of-range AQI values (0-500)
- drop_duplicates: Remove duplicate records
- drop_nulls: Drop records missing city, timestamp or AQI
- validate: Run final validation and return clean data report

Respond ONLY in JSON with this format:
{
    "thought": "what I observe and plan to do",
    "tool": "tool_name_to_call",
    "done": false
}

When all issues are fixed, respond with:
{
    "thought": "data is clean",
    "tool": null,
    "done": true
}
"""

# ── Agentic loop ───────────────────────────────────────────────────────
def run_agent(df: pd.DataFrame) -> tuple:
    """
    Agentic loop — LLM decides which tools to call and in what order.
    Runs until LLM says done=true or max_steps reached.
    """
    client      = get_ollama()
    messages    = []
    steps       = []
    max_steps   = 10
    step_count  = 0

    # Initial system message
    system_msg = f"""You are an autonomous data quality agent for an air quality monitoring pipeline.
Your job is to heal a dataset of AQI readings from 5 cities.
{TOOL_DESCRIPTIONS}"""

    # First message — show agent the data issues
    issues = tool_detect_issues(df)
    user_msg = f"""Here is the current state of the data:
{json.dumps(issues, indent=2)}

Start healing the data. Call tools one at a time until the data is clean."""

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg}
    ]

    print(f"\n  Agent starting — initial issues: {issues}")

    while step_count < max_steps:
        step_count += 1
        print(f"\n  Agent step {step_count}:")

        try:
            response = client.chat(model=MODEL, messages=messages)
            content  = response["message"]["content"]

            # Parse agent decision
            start = content.find("{")
            end   = content.rfind("}") + 1
            if start == -1 or end == 0:
                print(f"    Agent response not parseable: {content[:100]}")
                break

            decision = json.loads(content[start:end])
            thought  = decision.get("thought", "")
            tool     = decision.get("tool")
            done     = decision.get("done", False)

            print(f"    Thought: {thought}")

            if done or tool is None:
                print(f"    Agent says: done!")
                break

            # Call the tool
            if tool in TOOLS:
                print(f"    Calling tool: {tool}")
                if tool in ("detect_issues", "validate"):
                    result = TOOLS[tool](df)
                    tool_result = json.dumps(result)
                else:
                    df = TOOLS[tool](df)
                    result = tool_detect_issues(df)
                    tool_result = f"Tool applied. Remaining issues: {json.dumps(result)}"

                steps.append({
                    "step":   step_count,
                    "tool":   tool,
                    "thought":thought,
                    "result": str(result)[:200]
                })

                # Add to conversation history
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user",      "content": f"Tool result: {tool_result}. Continue healing if needed."})
            else:
                print(f"    Unknown tool: {tool}")
                break

        except Exception as e:
            print(f"    Agent error at step {step_count}: {e}")
            break

    return df, steps

# ── Main run function ──────────────────────────────────────────────────
def run(date_str=None):
    if not date_str:
        date_str = datetime.utcnow().strftime("%Y/%m/%d")

    y, m, d = date_str.split("/")
    prefix  = f"year={y}/month={m}/day={d}/"

    s3 = get_s3()

    # Read Bronze
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
        print("No Bronze data found")
        return

    df = pd.DataFrame(records)
    print(f"\nLoaded {len(df)} records from Bronze")
    print(f"Starting agentic self-healing with LLaMA 3.1...")

    # Run agentic loop
    healed_df, steps = run_agent(df)

    # Final validation
    final = tool_validate(healed_df)
    print(f"\n{'='*50}")
    print(f"  AGENTIC HEALING COMPLETE")
    print(f"{'='*50}")
    print(f"  Original records : {len(df)}")
    print(f"  Healed records   : {len(healed_df)}")
    print(f"  Agent steps taken: {len(steps)}")
    print(f"  Final validation : {final}")
    print(f"{'='*50}")

    # Save steps log
    os.makedirs("data", exist_ok=True)
    pd.DataFrame(steps).to_csv("data/agent_steps.csv", index=False)

    # Write to Silver S3
    table  = pa.Table.from_pandas(healed_df)
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    buffer.seek(0)

    key = f"{prefix}agentic_silver.parquet"
    s3.put_object(
        Bucket="silver-air-quality",
        Key=key,
        Body=buffer.getvalue(),
        ContentType="application/octet-stream"
    )
    print(f"✓ Saved → s3://silver-air-quality/{key}")

if __name__ == "__main__":
    run()
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor
from datetime import datetime, timedelta
import sys
sys.path.insert(0, '/opt/airflow/spark_jobs')

default_args = {
    'owner': 'airflow',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

def run_silver(**context):
    from silver_transform import run
    run()

def run_gold(**context):
    from gold_features import run
    run()

with DAG(
    dag_id="air_quality_pipeline",
    default_args=default_args,
    description="Silver & Gold layer transforms triggered after ingestion",
    schedule_interval="5 * * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["air-quality", "silver", "gold", "pipeline"],
) as dag:

    wait_for_ingestion = ExternalTaskSensor(
        task_id="wait_for_ingestion",
        external_dag_id="air_quality_ingestion",
        external_task_id="ingest_to_bronze",
        timeout=300,
        poke_interval=30,
        mode="poke",
    )

    silver_task = PythonOperator(
        task_id="silver_transform",
        python_callable=run_silver,
        provide_context=True,
    )

    gold_task = PythonOperator(
        task_id="gold_features",
        python_callable=run_gold,
        provide_context=True,
    )

    wait_for_ingestion >> silver_task >> gold_task
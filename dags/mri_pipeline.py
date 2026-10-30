from datetime import timedelta
from datetime import datetime
from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator
import airflow_config as c


with DAG(
    dag_id=c.DAG_ID,
    start_date=c.START_DATE,
    schedule=c.SCHEDULE,
    catchup=c.CATCHUP,
    max_active_runs=c.MAX_ACTIVE_RUNS,
    default_view=c.DEFAULT_VIEW,
    tags=c.TAGS,
    default_args=c.DEFAULT_ARGS
) as dag:
    pass
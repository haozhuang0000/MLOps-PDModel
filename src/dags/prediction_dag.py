from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from src.mlops.pipelines import prediction_service

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../..', '.env'))

default_args = {
    'owner': 'airflow',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='daily_prediction_service',
    default_args=default_args,
    description='Run prediction service daily',
    start_date=datetime(2025, 7, 14),
    schedule_interval='0 10 * * *',
    catchup=False,
) as dag:

    def run_prediction():
        from datetime import datetime
        task_date = datetime.today().strftime('%Y-%m-%d')
        prediction_service(2, 'LGBClassifier_Multiclass_CN', task_date)

    run_task = PythonOperator(
        task_id='run_prediction_service',
        python_callable=run_prediction
    )

    run_task
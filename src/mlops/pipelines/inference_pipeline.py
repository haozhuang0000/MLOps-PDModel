from src.mlops.steps_deployment import (
    wait_for_files,
    load_data,
    predict,
    save_mysql,
    post_statement
)
from datetime import datetime
from zenml.config import ResourceSettings
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
from dotenv import dotenv_values
env_vars = dotenv_values(".env")
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
requirements_path = os.path.join(current_dir, "../../../requirements.txt")
docker_settings = DockerSettings(requirements=requirements_path, apt_packages=["git", "libgomp1"], environment=env_vars)
@pipeline(
    enable_cache=False, 
    settings={
    "docker": docker_settings, 
    "resources": ResourceSettings(cpu_count=5, gpu_count=4, memory="24GB"),
    }
)
def prediction_service(econ:int, model_name:str, task_date: str):

    x_path, y_path, alert_signal = wait_for_files(econ=econ)
    post_statement(alert_signal, 'wait_for_files')

    X, y, alert_signal = load_data(x_path, y_path)
    post_statement(alert_signal, 'load_data')

    df, alert_signal = predict(model_name, X)
    post_statement(alert_signal, 'predict')

    alert_signal = save_mysql(df, econ, task_date)
    post_statement(alert_signal, 'save_mysql')

if __name__ == "__main__":
    task_date = datetime.today().strftime('%Y-%m-%d')
    prediction_service(2, 'LGBClassifier_Multiclass_CN', task_date)
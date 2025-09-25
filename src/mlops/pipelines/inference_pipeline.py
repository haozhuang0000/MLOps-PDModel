from src.mlops.steps_deployment import (
    wait_for_files,
    load_data,
    predict,
    save_mysql,
    post_statement
)
from datetime import datetime
from dotenv import dotenv_values
env_vars = dotenv_values(".env")
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
requirements_path = os.path.join(current_dir, "../../../requirements.txt")

def prediction_service(econ:int, model_name:str, task_date: str):

    x_path, y_path, cripd_path, cripoe_path, alert_signal, datadate = wait_for_files(econ=econ)
    # post_statement(alert_signal, 'wait_for_files')

    X, y, cripred, alert_signal = load_data(x_path, y_path, cripd_path, cripoe_path)
    # post_statement(alert_signal, 'load_data')

    df, alert_signal = predict(model_name, X, datadate)
    # post_statement(alert_signal, 'predict')

    alert_signal = save_mysql(df, y, cripred,econ, task_date)
    # post_statement(alert_signal, 'save_mysql')

if __name__ == "__main__":
    task_date = datetime.today().strftime('%Y%m%d')
    prediction_service(2, 'LGBClassifier_Multiclass_CN', task_date)
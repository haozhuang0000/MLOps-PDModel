from src.mlops.logger.utils.logger import Log
from src.mlops.evaluation import Evaluator, MSE, RMSE, R2Score
# from src.mlops.materializer import CSMaterializer
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, Dict, List, Any, Union
from typing_extensions import Annotated
from tqdm import tqdm
import os
import pandas as pd
from zenml import step
from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker

def evaluate_for_y(df_X_test: pd.DataFrame,
                   df_y_true: pd.Series,
                   mlflow_model_name: str,
                   mlflow_model_run_id: str):
    try:
        mse = Evaluator(strategy=MSE()).evaluate(df_X_test, df_y_true, mlflow_model_name, mlflow_model_run_id)
        rmse = Evaluator(strategy=RMSE()).evaluate(df_X_test, df_y_true, mlflow_model_name, mlflow_model_run_id)
        r2score = Evaluator(strategy=R2Score()).evaluate(df_X_test, df_y_true, mlflow_model_name, mlflow_model_run_id)
    except:
        print('groundtrue of Y is NAs')
        pass

def evaluate_for_model(df_test: pd.DataFrame, df_y_true: pd.DataFrame, y_and_model_info: Dict, id_bb_unique: str):

    for y, mlflow_model_info in y_and_model_info.items():
        df_X_test = df_test.drop(columns=[y])
        df_y_true_in = df_y_true[y]
        mlflow_model_name = mlflow_model_info["mlflow_model_name"]
        mlflow_model_run_id = mlflow_model_info["mlflow_model_run_id"]
        evaluate_for_y(df_X_test, df_y_true_in, mlflow_model_name, mlflow_model_run_id)


def evaluate_for_company(id_bb_unique: str, single_company_df_dict: Dict) -> None:

    # logger = Log(f"{os.path.basename(__file__)}").getlog()
    # for data_key, model_dict in single_company_df_dict.items():

    df_test = single_company_df_dict['data']['df_test_predby_arima']
    df_y_true = single_company_df_dict['data']['df_y_test']

    for model_name, y_and_model_info in single_company_df_dict['model'].items():
        # for y, mlflow_info in model_info.items():
        evaluate_for_model(df_test, df_y_true, y_and_model_info, id_bb_unique)

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(training_results: Dict[str, Any]):

    for id_bb_unique, single_company_df_dict in training_results.items():
        evaluate_for_company(id_bb_unique, single_company_df_dict)
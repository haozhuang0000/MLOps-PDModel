from src.mlops.logger.utils.logger import Log
from src.mlops.evaluation import Evaluator, Accuracy, Precision, Recall, F1Score, AucRoc, ArCredit
from src.mlops.configs import Variables
from src.mlops.model import ModelLoader
# from src.mlops.materializer import CSMaterializer
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, Dict, List, Any, Union
from typing_extensions import Annotated
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import mlflow
from mlflow import register_model
from datetime import datetime
# from zenml.config import ResourceSettings
# from zenml import step
# from zenml.client import Client
# experiment_tracker = Client().active_stack.experiment_tracker

# def evaluate_for_y(df_X_test: pd.DataFrame,
#                    df_y_true: pd.Series,
#                    mlflow_model_name: str,
#                    mlflow_model_run_id: str):
#     try:
#         mse = Evaluator(strategy=MSE()).evaluate(df_X_test, df_y_true, mlflow_model_name, mlflow_model_run_id)
#         rmse = Evaluator(strategy=RMSE()).evaluate(df_X_test, df_y_true, mlflow_model_name, mlflow_model_run_id)
#         r2score = Evaluator(strategy=R2Score()).evaluate(df_X_test, df_y_true, mlflow_model_name, mlflow_model_run_id)
#     except:
#         print('groundtrue of Y is NAs')
#         pass
#
# def evaluate_for_model(df_test: pd.DataFrame, df_y_true: pd.DataFrame, y_and_model_info: Dict, id_bb_unique: str):
#
#     for y, mlflow_model_info in y_and_model_info.items():
#         df_X_test = df_test.drop(columns=[y])
#         df_y_true_in = df_y_true[y]
#         mlflow_model_name = mlflow_model_info["mlflow_model_name"]
#         mlflow_model_run_id = mlflow_model_info["mlflow_model_run_id"]
#         evaluate_for_y(df_X_test, df_y_true_in, mlflow_model_name, mlflow_model_run_id)
#
#
# def evaluate_for_company(id_bb_unique: str, single_company_df_dict: Dict) -> None:
#
#     # logger = Log(f"{os.path.basename(__file__)}").getlog()
#     # for data_key, model_dict in single_company_df_dict.items():
#
#     df_test = single_company_df_dict['data']['df_test_predby_arima']
#     df_y_true = single_company_df_dict['data']['df_y_test']
#
#     for model_name, y_and_model_info in single_company_df_dict['model'].items():
#         # for y, mlflow_info in model_info.items():
#         evaluate_for_model(df_test, df_y_true, y_and_model_info, id_bb_unique)

# @step(experiment_tracker=experiment_tracker.name, settings={"resources": ResourceSettings(cpu_count=20, gpu_count=4, memory="128GB")})
def evaluate_model(final_test:pd.DataFrame,
                   training_results: Dict[str, Any],
                   average = 'macro',
                   method = 'classifier') -> Annotated[Dict[str, Any], "ml_result"]:
    X_test = final_test[Variables.x_cols_to_process_cn]
    y_test = final_test['Y']
    output_dict = {}
    model_info_dict = {}
    for model, model_results in training_results.items():
        mlflow_model_name = model_results["mlflow_model_name"]
        mlflow_model_run_id = model_results["mlflow_model_run_id"]

        # ==================== <START> Model inference ====================
        model, model_uri = ModelLoader.load_model(mlflow_model_name, mlflow_model_run_id)

        # Predict class probabilities or labels
        if method == 'classifier':
            y_proba = model.predict_proba(X_test)
        else:
            y_proba = model.predict(X_test)

        # If the model outputs probabilities, use argmax to get class labels
        if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
            y_pred = np.argmax(y_proba, axis=1)
        # ==================== <END> Model inference ====================

        ## Log to Mlflow
        Evaluator(strategy=Accuracy()).evaluate(y_pred, y_proba, y_test, mlflow_model_name, mlflow_model_run_id, average=None)
        Evaluator(strategy=Precision()).evaluate(y_pred, y_proba, y_test, mlflow_model_name, mlflow_model_run_id, average=average)
        Evaluator(strategy=Recall()).evaluate(y_pred, y_proba, y_test, mlflow_model_name, mlflow_model_run_id, average=average)
        Evaluator(strategy=F1Score()).evaluate(y_pred, y_proba, y_test, mlflow_model_name, mlflow_model_run_id, average=average)
        Evaluator(strategy=ArCredit()).evaluate(y_pred, y_proba, y_test, mlflow_model_name, mlflow_model_run_id, average=average)
        auc = Evaluator(strategy=AucRoc()).evaluate(y_pred, y_proba, y_test, mlflow_model_name, mlflow_model_run_id, average=average)
        metrics_type_name = AucRoc().__class__.__name__

        output_dict[mlflow_model_name] = {
            'mlflow_model_run_id': mlflow_model_run_id,
            'y_test': y_test,
            'y_pred_label': y_pred,
            'y_pred_proba': y_proba,
        }

        try:
            register_model_name = mlflow_model_name + f"_Multiclass_{training_results['cali_group']}"
        except:
            ## Todo：remove testing
            register_model_name = mlflow_model_name + f"_Multiclass_CN"
        result = mlflow.register_model(model_uri=model_uri, name=register_model_name)

        # metrics_dict = {
        #     'metrics_type_name': auc
        # }
        #
        # model_dict = {
        #     'model_type_name': mlflow_model_name,
        #     'version': register_model_name.version
        # }
        model_info_dict[mlflow_model_name] = {
            'metrics': {f'{metrics_type_name}': auc},
            'version': result.version,
            'train_date': datetime.today().date()
        }

    return output_dict, model_info_dict

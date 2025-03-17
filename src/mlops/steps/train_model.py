from src.mlops.logger.utils.logger import Log
from src.mlops.training import Trainer, MultivariateTraining
from src.mlops.model import RFRegression, LGBRegression, H2OAuto
from src.mlops.configs import Variables
# from src.mlops.materializer import CSMaterializer
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, Dict, List, Any, Union
from typing_extensions import Annotated
from tqdm import tqdm
import os
import pandas as pd
import pickle
import h2o
from zenml import step
from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker

# @step(experiment_tracker=experiment_tracker.name)
# def train_model(companyID_df_postConstru_dict: Dict):
#     logger = Log(f"{os.path.basename(__file__)}").getlog()
#     y_cols = Variables().y_cols
#     models = [LGBRegression(), RFRegression()]
#     results = {}
#     for id_bb_unique, single_company_df_dict in tqdm(companyID_df_postConstru_dict.items(), desc="Processing companies"):
#         company_results = {"data": single_company_df_dict, "model": {}}
#         for model in models:
#             model_name = model.__class__.__name__
#             model_results = {}
#             for y in y_cols:
#                 logger.info(f"Training model: **{model_name}** - company_id: **{id_bb_unique}** - y: **{y}**")
#                 trained_model, mlflow_model_name, mlflow_model_url = Trainer(strategy=MultivariateTraining()).run_training(
#                     single_company_df_dict=single_company_df_dict,
#                     id_bb_unique=id_bb_unique,
#                     y=y,
#                     model=model)
#
#                 model_results[y] = {
#                     "model": trained_model,
#                     "mlflow_model_name": mlflow_model_name,
#                     "mlflow_model_url": mlflow_model_url
#                 }
#             company_results["model"][model_name] = model_results
#         results[id_bb_unique] = company_results
#     return results

def train_for_company(id_bb_unique, single_company_df_dict):

    logger = Log(f"{os.path.basename(__file__)}").getlog()
    y_cols = Variables().y_cols
    # models = [H2OAuto(), LGBRegression(), RFRegression()]
    models = [LGBRegression(), RFRegression()]
    company_results = {"data": single_company_df_dict, "model": {}}
    for model in models:
        model_name = model.__class__.__name__
        model_results = {}
        for y in y_cols:
            logger.info(f"Training model: **{model_name}** - company_id: **{id_bb_unique}** - y: **{y}**")
            trained_model, mlflow_model_name, mlflow_model_run_id = Trainer(
                strategy=MultivariateTraining()).run_training(
                single_company_df_dict=single_company_df_dict,
                id_bb_unique=id_bb_unique,
                y=y,
                model=model)

            model_results[y] = {
                "model": trained_model,
                "mlflow_model_name": mlflow_model_name,
                "mlflow_model_run_id": mlflow_model_run_id
            }
        company_results["model"][model_name] = model_results
    return id_bb_unique, company_results

@step(experiment_tracker=experiment_tracker.name)
def train_model(companyID_df_postConstru_dict: Dict):
    results = {}
    companyID_df_postConstru_dict = {key: companyID_df_postConstru_dict[key] for key in list(companyID_df_postConstru_dict.keys())[:10]}
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/output'))
    file_path = os.path.join(base_dir, "training_results_auto.pkl")
    # Check if the file exists
    if os.path.exists(file_path):
        # Load the existing file
        with open(file_path, "rb") as file:
            results = pickle.load(file)
        print(f"File '{file_path}' loaded.")
    else:
        with ProcessPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(train_for_company, id_bb_unique, single_company_df_dict)
                for id_bb_unique, single_company_df_dict in companyID_df_postConstru_dict.items()
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing companies"):
                id_bb_unique, company_results = future.result()
                results[id_bb_unique] = company_results
        # for id_bb_unique, single_company_df_dict in tqdm(companyID_df_postConstru_dict.items(),
        #                                                  desc="Processing companies"):
        #     id_bb_unique, company_results = train_for_company(id_bb_unique, single_company_df_dict)
        #     results[id_bb_unique] = company_results
        with open(file_path, "wb") as file:
            pickle.dump(results, file)

    return results
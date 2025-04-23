from src.mlops.data_loading import DataLoader
from src.mlops.logger import LoggerDescriptor
# from src.mlops.materializer import CSMaterializer
from typing import Tuple, Dict
import os
import pickle
import pandas as pd
from zenml.config import ResourceSettings
from zenml import step
from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker

# @step(experiment_tracker=experiment_tracker.name)
# def load_data(data_path, comp_path) -> Tuple[pd.DataFrame, list, pd.DataFrame]:
#
#     df_annual_sorted_after_2000, industry_mappings, df_company_info = DataLoader().load_data(
#         data_path=data_path, comp_path=comp_path
#     )
#     return df_annual_sorted_after_2000, industry_mappings, df_company_info
#
# @step(experiment_tracker=experiment_tracker.name)
# def load_intermediate_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
#     # Todo: remove all data reading and dumping - this is for development purpose
#     df = pd.read_csv('../../../data/output/union_processed_imputed_expanded_80_20.csv')
#     df_ground_truth = pd.read_csv('../../../data/output/union_processed_groundtruth_expanded_80_20.csv')
#     return df, df_ground_truth
#
# @step(experiment_tracker=experiment_tracker.name)
# def load_intermediate_training_data() -> Dict:
#     # Todo: remove all data reading and dumping - this is for development purpose
#     base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/output'))
#     file_path = os.path.join(base_dir, "training_results.pkl")
#     # Check if the file exists
#     if os.path.exists(file_path):
#         # Load the existing file
#         with open(file_path, "rb") as file:
#             training_results = pickle.load(file)
#         print(f"File '{file_path}' loaded.")
#     return training_results

@step(enable_cache=False, experiment_tracker=experiment_tracker.name, settings={"resources": ResourceSettings(cpu_count=20, gpu_count=4, memory="128GB")})
def load_data(data_path) -> pd.DataFrame:
    print('loading data')
    df = DataLoader.load_data(
        data_path=data_path
    )
    return df
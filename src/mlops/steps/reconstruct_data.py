from src.mlops.data_reconstruction import DataReconstructor, BBGDataReconstructionTS
from src.mlops.logger.utils.logger import Log
from src.mlops.model import AR, ARIMA
# from src.mlops.materializer import CSMaterializer
from typing import Tuple, Union, Dict, Any
from typing_extensions import Annotated
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
import pickle
import os
from zenml import step
from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker

def process_single_company(id_bb_unique,
                           single_company_df_dict,
                           model,
                           modify_dict_type):
    """Worker function to process a single company's data."""
    return BBGDataReconstructionTS().handle_data(id_bb_unique,
                                                 single_company_df_dict,
                                                 model,
                                                 modify_dict_type)

def parallel_process(args_list, model_name, max_workers):
    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_company, *args)
            for args in args_list
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Running {model_name} Model"):
            id_bb_unique, data = future.result()
            results[id_bb_unique] = data
    return results

@step(experiment_tracker=experiment_tracker.name)
def reconstruct_data(companyID_df_dict: dict) -> Dict[str, Dict[str, Any]]:
    logger = Log(f"{os.path.basename(__file__)}").getlog()
    # logger.info(f"start data reconstruction - model: **{model.__class__.__name__}**")
    # companyID_df_postConstru_dict = DataReconstructor(strategy=BBGDataReconstructionTS()).handle_data(companyID_df_dict=companyID_df_dict,
    #                                                                                                   model=model)
    # Todo: remove all data reading and dumping - this is for development purpose
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/output'))
    file_path = os.path.join(base_dir, "companyID_df_postConstru_dict.pkl")
    # Check if the file exists
    if os.path.exists(file_path):
        # Load the existing file
        with open(file_path, "rb") as file:
            companyID_df_postConstru_dict = pickle.load(file)
        print(f"File '{file_path}' loaded.")
    else:
        ## ----------------------------------- AR ----------------------------------- ##
        logger.info(f"start data reconstruction - model: **{AR().__class__.__name__}**")
        args_list_ar = [
            (id_bb_unique, single_company_df_dict, AR(), 'create')
            for id_bb_unique, single_company_df_dict in companyID_df_dict.items()
        ]

        companyID_df_postConstru_ar_dict = parallel_process(args_list_ar, AR().__class__.__name__, 10)

        ## ----------------------------------- ARIMA ----------------------------------- ##
        logger.info(f"start data reconstruction - model: **{ARIMA().__class__.__name__}**")
        args_list_arima = [
            (id_bb_unique, single_company_df_dict, ARIMA(), 'append')
            for id_bb_unique, single_company_df_dict in companyID_df_postConstru_ar_dict.items()
        ]
        companyID_df_postConstru_dict = parallel_process(args_list_arima, ARIMA().__class__.__name__, 10)

        with open(file_path, "wb") as file:
            pickle.dump(companyID_df_postConstru_dict, file)
        print(f"File '{file_path}' created and saved.")

    return companyID_df_postConstru_dict
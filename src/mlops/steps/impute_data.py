from src.mlops.logger.utils.logger import Log
from src.mlops.configs import XYVariables
from src.mlops.data_imputation import DataImputer, BBGDataMVDetecter, BBGDataMVImputer
# from src.mlops.materializer import CSMaterializer
from typing import Tuple, List
from typing_extensions import Annotated
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from zenml import step
from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def impute_data(train_df_list: List[pd.DataFrame],
                ground_truth_list: List[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger = Log(f"{os.path.basename(__file__)}").getlog()
    train_df_list_imputed = []
    ground_truth_list_unchanged = []
    logger.info('1. replaces columns where more than threshold of values are missing with 0 '
                '2. imputes missing values from training data')
    for train_sub_df, test_sub_df in tqdm(zip(train_df_list, ground_truth_list), total=len(train_df_list),
                                          desc="Processing 1. detect missing value 2. impute missing value"):

        # Store ground truth for the test set
        ground_truth_list_unchanged.append(test_sub_df.copy())

        # Set specified columns in test_sub_df to np.nan
        test_sub_df[XYVariables().X_SELECTED] = np.nan

        # Step 1: Drop columns with high missing values marked as np.nan
        df = DataImputer(strategy=BBGDataMVDetecter()).handle_data(train_sub_df)

        # Step 2: Handle missing values for each id_bb_unique
        df_processed = DataImputer(strategy=BBGDataMVImputer()).handle_data(df)

        # Combine processed training_pipeline data with modified test data
        df_train = pd.concat([df_processed, test_sub_df], ignore_index=True)

        train_df_list_imputed.append(df_train)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/output'))

    df_combined_imputed = pd.concat(train_df_list_imputed, ignore_index=True)
    df_combined_imputed.to_csv(os.path.join(base_dir, 'union_processed_imputed_expanded_80_20.csv'), index=False)

    df_ground_truth = pd.concat(ground_truth_list_unchanged, ignore_index=True)
    df_ground_truth.to_csv(os.path.join(base_dir, 'union_processed_groundtruth_expanded_80_20.csv'), index=False)


    return df_combined_imputed, df_ground_truth






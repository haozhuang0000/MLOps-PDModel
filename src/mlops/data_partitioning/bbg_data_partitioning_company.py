from src.mlops.abstractions import DataPreprocessingStrategyABC
import pandas as pd
from typing import List, Tuple, Dict
from tqdm import tqdm

class BBGDataPartitioningCompany(DataPreprocessingStrategyABC):

    def handle_data(self, df: pd.DataFrame, df_ground_truth: pd.DataFrame) -> dict:

        ## {'ID_BB_UNIQUE': sub_df} -> ID_BB_UNIQUE as key, its sub df as value
        companyID_df_dict = {}

        for id_bb_uniquq, sub_df in tqdm(df.groupby('ID_BB_UNIQUE'),
                                         desc=f"running data partitioning - calling: {BBGDataPartitioningCompany().__class__.__name__}"):
            nan_rows = sub_df[sub_df.isnull().any(axis=1)].index.tolist()
            sub_df_train = sub_df.dropna()
            df_test = sub_df.loc[nan_rows]
            companyID_specific_dict = {
                'df_preconstru': sub_df,
                'df_train': sub_df_train,
                'df_test': df_test,
                'nan_rows': nan_rows,
            }
            companyID_df_dict[id_bb_uniquq] = companyID_specific_dict

        for id_bb_uniquq, sub_df in df_ground_truth.groupby('ID_BB_UNIQUE'):

            companyID_df_dict[id_bb_uniquq]['df_ground_truth'] = sub_df

        return companyID_df_dict






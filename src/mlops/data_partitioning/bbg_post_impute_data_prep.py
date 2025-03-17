from src.mlops.abstractions import DataPreprocessingStrategyABC
from typing import Union, Tuple, List
import pandas as pd
import numpy as np
from tqdm import tqdm

class BBGPostImputeDataPrep(DataPreprocessingStrategyABC):

    def handle_data(self, df: pd.DataFrame, df_ground_truth: pd.DataFrame) \
            -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list]:

        df['Year'] = pd.to_datetime(df['Year'], format='%Y')
        df_x = df[self.x_cols_to_process]
        nan_rows = df[df.isnull().any(axis=1)].index.tolist()
        df_industry_train = df.drop(nan_rows)
        df_test = df.loc[nan_rows]

        return df, df_x, df_ground_truth, \
                df_industry_train, df_test, nan_rows


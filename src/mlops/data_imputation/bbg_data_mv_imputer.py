from typing import Union
import pandas as pd
from tqdm import tqdm
from src.mlops.abstractions import DataImputationStrategyABC
from src.mlops.data_imputation.data_imputer_helper import BBGDataMVImputerHelper

class BBGDataMVImputer(DataImputationStrategyABC):

    def handle_data(self, df: pd.DataFrame) -> pd.DataFrame:

        for col in df.columns:
            df[col] = BBGDataMVImputerHelper()._run(series=df[col])

        return df





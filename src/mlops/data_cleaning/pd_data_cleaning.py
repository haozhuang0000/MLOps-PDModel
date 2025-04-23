from src.mlops.abstractions import DataPreprocessingStrategyABC
from typing import Union
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

class PDDataCleaning(DataPreprocessingStrategyABC):

    def handle_data(self, df: pd.DataFrame):

        # self.logger.info('Droping missing values')
        # df = df.drop(
        #     [
        #         "CompNo",
        #         "yyyy",
        #         "mm",
        #         "StkIndx"
        #     ],
        #     axis=1
        # )

        # df = df.dropna()
        return df
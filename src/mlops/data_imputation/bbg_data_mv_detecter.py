from src.mlops.abstractions import DataImputationStrategyABC
from typing import Union
import pandas as pd

class BBGDataMVDetecter(DataImputationStrategyABC):

    def handle_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replaces columns where more than 'threshold' fraction of values are missing
        with np.nan.

        Parameters:
            df (pd.DataFrame): The input DataFrame.
            threshold (float): The fraction of allowed missing values. Columns with
                               a higher fraction of missing values will be replaced
                               with np.nan.

        Returns:
            pd.DataFrame: The DataFrame with specified columns replaced by np.nan.
        """
        # Calculate the fraction of missing values in each column
        missing_fraction = df.isna().mean()

        # Identify columns where the missing fraction exceeds the threshold
        cols_to_mark = missing_fraction[missing_fraction > self.threshold].index

        # Replace entire columns with np.nan
        df[cols_to_mark] = 0

        return df
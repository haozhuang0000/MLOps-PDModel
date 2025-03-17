from src.mlops.abstractions import DataImputationStrategyABC, DataImputationHelperStrategyABC
from typing import Union
import pandas as pd

class DataImputer:

    def __init__(self, strategy: DataImputationStrategyABC):

        self.strategy = strategy

    def handle_data(self, df: pd.DataFrame) -> pd.DataFrame:

        return self.strategy.handle_data(df)

class DataImputationHelper:

    def __init__(self, strategy: DataImputationHelperStrategyABC):

        self.strategy = strategy

    def handle_data(self, series: pd.Series) -> pd.Series:

        return self.strategy.handle_data(series)
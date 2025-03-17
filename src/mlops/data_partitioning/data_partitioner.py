from src.mlops.abstractions import DataPreprocessingStrategyABC
from typing import Union
import pandas as pd

class DataPartitioner:

    def __init__(self, strategy: DataPreprocessingStrategyABC):
        self.strategy = strategy

    def handle_data(self, **kwargs) -> Union[pd.DataFrame, pd.Series, list]:
        return self.strategy.handle_data(**kwargs)
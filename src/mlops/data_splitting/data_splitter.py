from src.mlops.abstractions import DataPreprocessingStrategyABC
from typing import Union, Tuple, List
import pandas as pd

class DataSplitter:

    def __init__(self, strategy: DataPreprocessingStrategyABC):

        self.strategy = strategy

    def handle_data(self, **kwargs) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:

        return self.strategy.handle_data(**kwargs)
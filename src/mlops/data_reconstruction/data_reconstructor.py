from src.mlops.abstractions import DataReconstructionStrategyABC
from typing import Union
import pandas as pd

class DataReconstructor:

    def __init__(self, strategy: DataReconstructionStrategyABC):
        self.strategy = strategy

    def handle_data(self, **kwargs) -> Union[pd.DataFrame, pd.Series, list]:
        return self.strategy.handle_data(**kwargs)
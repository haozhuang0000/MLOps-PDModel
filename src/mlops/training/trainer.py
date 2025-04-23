from src.mlops.abstractions import TrainingABC
from typing import Union
import pandas as pd

class Trainer:

    def __init__(self, strategy: TrainingABC):
        self.strategy = strategy

    def run_training(self, **kwargs) -> Union[pd.DataFrame, pd.Series, list]:
        return self.strategy.run_training(**kwargs)
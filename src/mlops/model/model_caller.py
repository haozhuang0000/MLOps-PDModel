from src.mlops.abstractions import ModelABC, UnivariateTSModelABC
from lightgbm import Booster
from sklearn.base import RegressorMixin
from typing import Union, Tuple, List
import numpy as np
import pandas as pd

class ModelCaller:

    def __init__(self, strategy: Union[ModelABC, UnivariateTSModelABC]):
        self.strategy = strategy

    def train(self, **kwargs) -> Union[Booster, RegressorMixin, str]:
        return self.strategy.train(**kwargs)
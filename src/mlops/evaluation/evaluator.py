from src.mlops.abstractions import EvaluationABC
from lightgbm import Booster
from sklearn.base import RegressorMixin
from typing import Union, Tuple, List
import numpy as np
import pandas as pd

class Evaluator(EvaluationABC):

    def __init__(self, strategy: EvaluationABC):
        self.strategy = strategy

    def evaluate(self,
                 y_pred: np.array,
                 y_proba: np.array,
                 y_test: pd.Series,
                 mlflow_model_name: str,
                 mlflow_model_run_id: str,
                 average: str) -> None:

        return self.strategy.evaluate(y_pred,
                                      y_proba,
                                      y_test,
                                      mlflow_model_name,
                                      mlflow_model_run_id,
                                      average)
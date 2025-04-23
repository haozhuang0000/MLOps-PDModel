from src.mlops.logger import LoggerDescriptor
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class EvaluationABC(ABC):
    """
    Abstract Class defining the strategy for evaluating model performance
    """

    logger = LoggerDescriptor()
    @abstractmethod
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass
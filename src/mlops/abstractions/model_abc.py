from src.mlops.logger import LoggerDescriptor
from src.mlops.configs import Variables
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import os
from typing import List, Tuple, Union
from sklearn.base import RegressorMixin
from lightgbm import Booster

class ModelABC(ABC, Variables):
    """
    Abstract base class for all models.
    """

    logger = LoggerDescriptor()
    forecast_points = None

    @abstractmethod
    def train(self, X_train: pd.DataFrame,
                    y_train: pd.Series,
                    id_bb_unique: str,
                    y: str) -> Union[RegressorMixin, Booster]:
        """
        Trains the model on the given data.

        Args:
            x_train: Training data
            y_train: Target data
        """
        pass

    # @abstractmethod
    # def optimize(self, trial, x_train, y_train, x_test, y_test):
    #     """
    #     Optimizes the hyperparameters of the model.
    #
    #     Args:
    #         trial: Optuna trial object
    #         x_train: Training data
    #         y_train: Target data
    #         x_test: Testing data
    #         y_test: Testing target
    #     """
    #     pass

class UnivariateTSModelABC(ABC):
    """
    Abstract base class for all models.
    """
    logger = LoggerDescriptor()
    @abstractmethod
    def train(self, training_points: pd.Series, forecast_points: int) -> list:
        """
        Trains the model on the given data.

        Args:
            x_train: Training data
            y_train: Target data
        """
        pass
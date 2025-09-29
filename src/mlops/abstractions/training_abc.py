from src.mlops.configs import Variables, TrainingVariables
from src.mlops.logger import LoggerDescriptor
from abc import ABC, abstractmethod
from typing import Dict
import numpy as np
import pandas as pd
import os

class TrainingABC(ABC, Variables, TrainingVariables):

    logger = LoggerDescriptor()
    @abstractmethod
    def run_training(self, companyID_df_postConstru_dict: Dict, x_reconstruction_type: str) -> None:
        """
        Executes the training process for a machine learning model.

        Parameters:
            df_train_x (pd.DataFrame): The input features for training.
            df_train_y (pd.DataFrame): The target labels for training.
            target_columns (str): The column name in df_train_y representing the target variable(s).
            id_bb_unique (str): A unique identifier, possibly for batch or block-based processing.
            x_reconstruction_type (str): Specifies the type of reconstruction for the input features
                                         (e.g., 'pca', 'autoencoder', etc.).

        Returns:
            pd.DataFrame: A DataFrame containing the training results or model outputs.
                          (Currently not implemented; returns `pass`.)
        """
        pass

# class TrainingHelperABC(ABC):
#
#     @abstractmethod
#     def run_training(self, df_train_x: pd.DataFrame, df_train_y: pd.Series) -> None:
#         """
#         Executes the training process for a machine learning model.
#
#         Parameters:
#             df_train_x (pd.DataFrame): The input features for training.
#             df_train_y (pd.DataFrame): The target labels for training.
#         """
#         pass
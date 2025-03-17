from src.mlops.logger import LoggerDescriptor
from src.mlops.configs import Variables
from abc import ABC, abstractmethod
from typing import Union
import pandas as pd

class DataPreprocessingStrategyABC(ABC, Variables):
    """
    Abstract Class defining strategy for handling data
    """
    logger = LoggerDescriptor()

    @abstractmethod
    def handle_data(self, **kwargs) -> Union[pd.DataFrame, pd.Series, list, dict]:
        pass

class DataImputationStrategyABC(ABC):
    """
    Abstract Class defining strategy for handling data
    """
    logger = LoggerDescriptor()
    threshold = 0.8

    @abstractmethod
    def handle_data(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class DataImputationHelperStrategyABC(ABC):
    logger = LoggerDescriptor()
    @abstractmethod
    def handle_data(self, series: pd.Series) -> pd.Series:
        pass

class DataReconstructionStrategyABC(ABC, Variables):
    logger = LoggerDescriptor()

    @abstractmethod
    def handle_data(self, companyID_df_dict: dict, **kwargs) -> dict:
        pass

from src.mlops.logger import LoggerDescriptor
from src.mlops.data_splitting import DataSplitter, BBGDataSplitter, PDDataSplitter
# from src.mlops.materializer import CSMaterializer
from typing import Tuple, List, Union, Dict
from typing_extensions import Annotated
import pandas as pd
from zenml.config import ResourceSettings
from zenml import step
from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker

# def split_data(df: pd.DataFrame) -> Tuple[
#                                     Annotated[pd.DataFrame, "X_train"],
#                                     Annotated[pd.DataFrame, "X_valid"],
#                                     Annotated[pd.DataFrame, "X_test"],
#                                     Annotated[pd.Series, "y_train"],
#                                     Annotated[pd.Series, "y_valid"],
#                                     Annotated[pd.Series, "y_test"],]:

@step(experiment_tracker=experiment_tracker.name, settings={"resources": ResourceSettings(cpu_count=20, gpu_count=4, memory="128GB")})
def split_data(df: pd.DataFrame) -> Tuple[
                                    Annotated[List, "kfolds_train_val"],
                                    Annotated[pd.DataFrame, "train_df"],
                                    Annotated[pd.DataFrame, "test_df"]]:

    train_val_splits, train_df, final_test = DataSplitter(strategy=PDDataSplitter()).handle_data(df=df)
    return train_val_splits, train_df, final_test
from src.mlops.logger import LoggerDescriptor
from src.mlops.data_splitting import DataSplitter, BBGDataSplitter
# from src.mlops.materializer import CSMaterializer
from typing import Tuple, List
from typing_extensions import Annotated
import pandas as pd
from zenml import step
from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def split_data(df_combined: pd.DataFrame) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:

    train_df_list, ground_truth_list = DataSplitter(strategy=BBGDataSplitter()).handle_data(
        df_combined=df_combined
    )
    return train_df_list, ground_truth_list
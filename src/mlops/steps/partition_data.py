from src.mlops.data_partitioning import DataPartitioner, BBGPostImputeDataPrep, BBGDataPartitioningCompany
# from src.mlops.materializer import CSMaterializer
from typing import Tuple, Dict, Any
from typing_extensions import Annotated
import pandas as pd
from zenml import step
from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def partition_data(df: pd.DataFrame, df_ground_truth: pd.DataFrame) \
        -> Dict[str, Dict[str, Any]]:

    df, df_x, df_ground_truth, df_industry_train, df_test, nan_rows = \
        DataPartitioner(strategy=BBGPostImputeDataPrep()).handle_data(df=df, df_ground_truth=df_ground_truth)

    companyID_df_dict = DataPartitioner(strategy=BBGDataPartitioningCompany()).handle_data(df=df, df_ground_truth=df_ground_truth)

    return companyID_df_dict
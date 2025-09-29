from src.mlops.logger import LoggerDescriptor
from src.mlops.data_cleaning import PDDataCleaning, DataCleaner
from typing import Tuple
from typing_extensions import Annotated
import pandas as pd
# from zenml.config import ResourceSettings
# from zenml import step
# from zenml.client import Client
# experiment_tracker = Client().active_stack.experiment_tracker

# @step(experiment_tracker=experiment_tracker.name, settings={"resources": ResourceSettings(cpu_count=20, gpu_count=4, memory="128GB")})
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    #
    ## ----------------------------- merge fs data with company info ----------------------------- ##
    # df_merged, industry_cols_to_merge = DataCleaner(strategy=BBGDataMerger()).handle_data(
    #     df=df,
    #     industry_mappings=industry_mappings,
    #     df_company_info=df_company_info
    # )
    #
    # df_combined, cols_to_process = DataCleaner(strategy=BBGDataUnion()).handle_data(
    #     df_merged=df_merged,
    #     industry_cols_to_merge=industry_cols_to_merge
    # )
    # # df_combined = pd.read_csv(r'../../../data/output/union_processed.csv')
    # df_combined = DataCleaner(strategy=BBGDataExpansion()).handle_data(df_combined=df_combined)
    # df_ticker_join = df_company_info[['ID_BB_UNIQUE', 'TICKER']]
    # df_combined = df_combined.merge(df_ticker_join, how='left', on='ID_BB_UNIQUE')

    df = DataCleaner(strategy=PDDataCleaning()).handle_data(df=df)
    return df


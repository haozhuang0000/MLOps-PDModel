from src.mlops.logger import LoggerDescriptor
from src.mlops.data_cleaning import DataCleaner, BBGDataMerger, BBGDataUnion, BBGDataExpansion
from typing import Tuple
from typing_extensions import Annotated
import pandas as pd
from zenml import step
from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def clean_data(df_annual_sorted_after_2000: pd.DataFrame,
               industry_mappings: list,
               df_company_info: pd.DataFrame) -> pd.DataFrame:
    #
    ## ----------------------------- merge fs data with company info ----------------------------- ##
    df_merged, industry_cols_to_merge = DataCleaner(strategy=BBGDataMerger()).handle_data(
        df_annual_sorted_after_2000=df_annual_sorted_after_2000,
        industry_mappings=industry_mappings,
        df_company_info=df_company_info
    )

    df_combined, cols_to_process = DataCleaner(strategy=BBGDataUnion()).handle_data(
        df_merged=df_merged,
        industry_cols_to_merge=industry_cols_to_merge
    )
    # df_combined = pd.read_csv(r'../../../data/output/union_processed.csv')
    df_combined = DataCleaner(strategy=BBGDataExpansion()).handle_data(df_combined=df_combined)
    df_ticker_join = df_company_info[['ID_BB_UNIQUE', 'TICKER']]
    df_combined = df_combined.merge(df_ticker_join, how='left', on='ID_BB_UNIQUE')
    return df_combined


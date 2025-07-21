# from zenml.config import DockerSettings
# from zenml.integrations.constants import MLFLOW
# from zenml.pipelines import pipeline
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
from src.mlops.steps import (
    load_data,
    clean_data,
    split_data, train_model,
    train_model,
    evaluate_model,
    evidently_eval_report,
)
# from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/input'))

# docker_settings = DockerSettings(required_integrations=[MLFLOW])
# @pipeline(enable_cache=False, settings={"docker": docker_settings})
def train_pipeline():
    # ### --------------------------- Data Preprocessing ------------------------ ###
    # df_annual_sorted_after_2000, industry_mappings, df_company_info = load_data(
    #     data_path=os.path.join(base_dir, 'union_ebitda_rev_cashflowfromoper_capex_merged_with_x_vars.csv'),
    #     comp_path=os.path.join(base_dir, 'Company Info.xlsx')
    # )
    #
    # df_combined = clean_data(df_annual_sorted_after_2000=df_annual_sorted_after_2000,
    #                          industry_mappings=industry_mappings,
    #                          df_company_info=df_company_info)
    # train_df_list, ground_truth_list = split_data(df_combined=df_combined)
    # df, df_ground_truth = impute_data(train_df_list=train_df_list, ground_truth_list=ground_truth_list)
    #
    # # Todo: remove all data reading and dumping - this is for development purpose
    # df, df_ground_truth = load_intermediate_data()
    # companyID_df_dict = partition_data(df, df_ground_truth)
    # companyID_df_postConstru_dict = reconstruct_data(companyID_df_dict)
    #
    # # Todo: ### --------------------------- Run univariate ts pipeline ------------------------ ###
    #
    # ### --------------------------- Run multi-variate ts pipeline ------------------------ ###
    # trained_results = train_model(companyID_df_postConstru_dict)
    # # trained_results = load_intermediate_training_data()
    # evaluated_results = evaluate_model(trained_results)

    df = load_data(data_path=os.path.join(base_dir, 'merged_output_1m.csv'))
    df = clean_data(df)
    # X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(df)
    train_val_splits, train_df, final_test = split_data(df)
    trained_results = train_model(train_val_splits, train_df)
    ml_result = evaluate_model(final_test, trained_results)
    evidently_eval_report(ml_result, train_df, final_test)

    # print(
    #     "Now run \n "
    #     f"    mlflow ui --backend-store-uri '{get_tracking_uri()}' --host 0.0.0.0\n"
    #     "To inspect your experiment runs within the mlflow UI.\n"
    #     "You can find your runs tracked within the `mlflow_example_pipeline`"
    #     "experiment. Here you'll also be able to compare the two runs.)"
    # )


if __name__ == "__main__":
    train_pipeline()
    pass
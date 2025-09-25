# from zenml.config import DockerSettings
# from zenml.integrations.constants import MLFLOW
# from zenml.pipelines import pipeline
from typing import Dict, List, Tuple
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
    save_training_info_into_mysql

)
# from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/input'))

# docker_settings = DockerSettings(required_integrations=[MLFLOW])
# @pipeline(enable_cache=False, settings={"docker": docker_settings})
def train_pipeline(cali_group=2, target_list: List[int] = None):

    for target in target_list:
        df = load_data(data_path=os.path.join(base_dir, 'merged_output_1m.csv'))
        df = clean_data(df)
        # X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(df)
        train_val_splits, train_df, final_test = split_data(df)
        trained_results = train_model(train_val_splits, train_df, cali_group=cali_group)
        ml_result, model_info_dict = evaluate_model(final_test, trained_results)
        save_training_info_into_mysql(model_info_dict=model_info_dict, econ_id=cali_group, target=target)
        evidently_eval_report(ml_result, train_df, final_test)

if __name__ == "__main__":
    # target_list = [i for i in range(len(1))]

    ## horizon of the pd
    target_list = [1]
    train_pipeline(cali_group=2, target_list=target_list)
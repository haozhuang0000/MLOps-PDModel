from src.mlops.evaluation_report import ArMetric
from src.mlops.data_loading import DataLoader
from evidently.future.report import Report
from evidently.future.report import Context
from evidently.future.datasets import Dataset
from evidently.future.datasets import DataDefinition
from evidently.future.datasets import MulticlassClassification, BinaryClassification
from evidently.future.metrics import *
from evidently.future.presets import *

from typing import Tuple, Dict, List, Any, Union
from typing_extensions import Annotated
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import mlflow
# from zenml.config import ResourceSettings
# from zenml import step
# from zenml.client import Client
# experiment_tracker = Client().active_stack.experiment_tracker

# @step(experiment_tracker=experiment_tracker.name, settings={"resources": ResourceSettings(cpu_count=20, gpu_count=4, memory="128GB")})
def evidently_eval_report(ml_results: Dict[str, Any], train_df: pd.DataFrame, test_df: pd.DataFrame):
    ## Set output dir for evidently report
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/output/evidently_ai'))
    os.makedirs(base_dir, exist_ok=True)

    ## Define evidently dataset definition
    data_def = DataDefinition(
        classification=[BinaryClassification(
            target="target",
            prediction_probas="class_1_prob",
        )]
    )

    # Define DataDefinition with class probabilities split into separate columns
    # data_def = DataDefinition(
    #     classification=[MulticlassClassification(
    #         target="target",
    #         prediction_labels="prediction",
    #         prediction_probas=["0", "1", "2"],
    #         labels={'0': "class_0", "1": "class_1", "2": "class_2"}
    #     )]
    # )

    ## Load PD Data for comparison
    ## Todo: change this temp path
    df_pd = DataLoader.load_windows_data(r"\\10.230.252.2\TeamData\Shared Temp\China_ML\combine_1m.csv")
    all_years = df_pd['yyyy'].unique()
    final_test_years = all_years[-3:]
    df_pd = df_pd[df_pd['yyyy'].isin(final_test_years)]
    df_pd = df_pd[['Y', 'pd_1m']].rename(columns={"pd_1m": "class_1_prob", "Y": "target"})
    df_pd["class_1_prob"] = df_pd["class_1_prob"]
    df_pd["target"] = df_pd["target"]
    df_pd["target"] = (df_pd["target"].astype(int) == 1).astype(int)

    pd_dataset = Dataset.from_pandas(df_pd, data_definition=data_def)

    ## Data summary
    dataset_train = Dataset.from_pandas(
        pd.DataFrame(train_df)
    )

    dataset_test = Dataset.from_pandas(
        pd.DataFrame(test_df)
    )
    report_train_data = Report([
        DataSummaryPreset()
    ])
    report_test_data = Report([
        DataSummaryPreset()
    ])
    train_data_report = report_train_data.run(dataset_train, None)
    test_data_report = report_test_data.run(dataset_test, None)

    for model_name, results in ml_results.items():

        mlflow_model_run_id = results['mlflow_model_run_id']
        y_pred_proba = results['y_pred_proba'] # shape: [n_samples, 3]
        y_pred_label = results['y_pred_label'] # shape: [n_samples,]
        y_test = results['y_test']  # shape: [n_samples,]

        # # Extract separate class probability columns
        proba_class_0 = y_pred_proba[:, 0]
        proba_class_1 = y_pred_proba[:, 1]
        proba_class_2 = y_pred_proba[:, 2]

        # Create the ML model DataFrame
        ml_model_df = pd.DataFrame({
            "target": y_test,
            "prediction": y_pred_label,
            "class_0_prob": proba_class_0,
            "class_1_prob": proba_class_1,
            "class_2_prob": proba_class_2
        })
        ml_model_df = ml_model_df[['target', 'class_1_prob']]
        ml_model_df["target"] = (ml_model_df["target"].astype(int) == 1).astype(int)
        # # Create the PD model DataFrame (reference data)
        # pd_model_df = pd.DataFrame({
        #     "target": y_test,
        #     "prediction": y_pred_label,
        #     "0": proba_class_0,
        #     "1": proba_class_1,
        #     "2": proba_class_2
        # })

        # Create Evidently Dataset objects
        ml_dataset = Dataset.from_pandas(ml_model_df, data_definition=data_def)

        report = Report([
            ClassificationPreset(),
            ArMetric(y_true="target", y_pred_probas_1="class_1_prob")
        ])
        my_eval = report.run(current_data=ml_dataset, reference_data=pd_dataset)

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/output/evidently_ai'))
        os.makedirs(base_dir, exist_ok=True)
        file_path = os.path.join(base_dir, f"{mlflow_model_run_id}_eval_report.html")
        train_data_report_file_path = os.path.join(base_dir, f"{mlflow_model_run_id}_train_data.html")
        test_data_report_file_path = os.path.join(base_dir, f"{mlflow_model_run_id}_test_data.html")
        my_eval.save_html(file_path)
        train_data_report.save_html(train_data_report_file_path)
        test_data_report.save_html(test_data_report_file_path)
        with mlflow.start_run(run_id=mlflow_model_run_id, nested=True):
            mlflow.log_artifact(file_path,"eval_report")
            mlflow.log_artifact(train_data_report_file_path, "train_data_summary")
            mlflow.log_artifact(test_data_report_file_path, "test_data_summary")


from src.mlops.evaluation_report import ArMetric
from evidently.future.report import Report
from evidently.future.report import Context
from evidently.future.datasets import Dataset
from evidently.future.datasets import DataDefinition
from evidently.future.datasets import MulticlassClassification
from evidently.future.metrics import *
from evidently.future.presets import *

from typing import Tuple, Dict, List, Any, Union
from typing_extensions import Annotated
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import mlflow
from zenml.config import ResourceSettings
from zenml import step
from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name, settings={"resources": ResourceSettings(cpu_count=20, gpu_count=4, memory="128GB")})
def evidently_eval_report(ml_results: Dict[str, Any]):

    for model_name, results in ml_results.items():

        mlflow_model_run_id = results['mlflow_model_run_id']
        y_pred_proba = results['y_pred_proba']  # shape: [n_samples, 3]
        y_pred_label = results['y_pred_label'].astype(str)  # shape: [n_samples,]
        y_test = results['y_test'].astype(str)  # shape: [n_samples,]

        # Extract separate class probability columns
        proba_class_0 = y_pred_proba[:, 0]
        proba_class_1 = y_pred_proba[:, 1]
        proba_class_2 = y_pred_proba[:, 2]

        # Create the ML model DataFrame
        ml_model_df = pd.DataFrame({
            "target": y_test,
            "prediction": y_pred_label,
            "0": proba_class_0,
            "1": proba_class_1,
            "2": proba_class_2
        })

        # Create the PD model DataFrame (reference data)
        pd_model_df = pd.DataFrame({
            "target": y_test,
            "prediction": y_pred_label,
            "0": proba_class_0,
            "1": proba_class_1,
            "2": proba_class_2
        })

        # Define DataDefinition with class probabilities split into separate columns
        data_def = DataDefinition(
            classification=[MulticlassClassification(
                target="target",
                prediction_labels="prediction",
                prediction_probas=["0", "1", "2"],
                labels={'0': "class_0", "1": "class_1", "2": "class_2"}
            )]
        )

        # Create Evidently Dataset objects
        ml_dataset = Dataset.from_pandas(ml_model_df, data_definition=data_def)
        pd_dataset = Dataset.from_pandas(pd_model_df, data_definition=data_def)

        report = Report([
            ClassificationPreset(),
            ArMetric(y_true="target", y_pred_probas_0="0", y_pred_probas_1="1", y_pred_probas_2="2")
        ])
        my_eval = report.run(current_data=ml_dataset, reference_data=pd_dataset)

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/output/evidently_ai'))
        os.makedirs(base_dir, exist_ok=True)
        file_path = os.path.join(base_dir, f"{mlflow_model_run_id}_eval_report.html")
        my_eval.save_html(file_path)
        with mlflow.start_run(run_id=mlflow_model_run_id, nested=True):
            mlflow.log_artifact(file_path,
                                "eval_report")

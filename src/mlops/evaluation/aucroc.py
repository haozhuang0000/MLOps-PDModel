from src.mlops.abstractions import EvaluationABC
from src.mlops.model import ModelLoader
from typing import Literal
import numpy as np
import pandas as pd
import os
import mlflow
from sklearn.metrics import roc_auc_score

class AucRoc(EvaluationABC):

    def evaluate(self,
                 y_pred: np.array,
                 y_proba: np.array,
                 y_test: pd.Series,
                 mlflow_model_name: str,
                 mlflow_model_run_id: str,
                 average: str = 'macro',
                 multi_class: str = 'ovr') -> float:

        # model, model_uri = ModelLoader.load_model(mlflow_model_name, mlflow_model_run_id)
        #
        # # Predict probabilities
        # if method == 'classifier':
        #     y_proba = model.predict_proba(X_test)
        # else:
        #     y_proba = model.predict(X_test)
        #
        # # Ensure y_test is numeric (for multi-class ROC AUC)
        # if isinstance(y_test.iloc[0], str):
        #     y_test = pd.factorize(y_test)[0]

        auc = roc_auc_score(y_test, y_proba, average=average, multi_class=multi_class)

        from src.mlops.evaluation import evaluate_outsample_ovo
        metric = 'roc_auc_score'
        auc_ovo_01 = evaluate_outsample_ovo(metric, y_test, y_proba, y_proba, class_a=0, class_b=1)
        auc_ovo_02 = evaluate_outsample_ovo(metric, y_test, y_proba, y_proba, class_a=0, class_b=2)

        # Enable autologging
        if "LGB" in mlflow_model_name:
            mlflow.lightgbm.autolog()
        elif "RF" in mlflow_model_name:
            mlflow.sklearn.autolog()

        # Log AUC
        with mlflow.start_run(run_id=mlflow_model_run_id, nested=True):
            mlflow.log_metric(f"outsample_{multi_class}_auc_{average}", auc)
            mlflow.log_metric(f"outsample_class0v1_auc", auc_ovo_01)
            mlflow.log_metric(f"outsample_class0v2_auc", auc_ovo_02)

        return auc
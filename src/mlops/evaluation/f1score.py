from src.mlops.abstractions import EvaluationABC
from src.mlops.model import ModelLoader
from typing import Literal
import numpy as np
import pandas as pd
import os
import mlflow
from sklearn.metrics import f1_score

class F1Score(EvaluationABC):

    def evaluate(self,
                 y_pred: np.array,
                 y_proba: np.array,
                 y_test: pd.Series,
                 mlflow_model_name: str,
                 mlflow_model_run_id: str,
                 average: str = 'macro') -> float:

        # model, model_uri = ModelLoader.load_model(mlflow_model_name, mlflow_model_run_id)
        #
        # # Predict probabilities
        # if method == 'classifier':
        #     y_proba = model.predict_proba(X_test)
        # else:
        #     y_proba = model.predict(X_test)
        #
        # # If model returns probabilities, convert to class predictions
        # if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
        #     y_pred = np.argmax(y_proba, axis=1)

        f1_ovr = f1_score(y_test, y_pred, average=average)

        from src.mlops.evaluation import evaluate_outsample_ovo
        metric = 'f1_score'
        f1_ovo_01 = evaluate_outsample_ovo(metric, y_test, y_pred, y_proba, class_a=0, class_b=1)
        f1_ovo_02 = evaluate_outsample_ovo(metric, y_test, y_pred, y_proba, class_a=0, class_b=2)

        # Enable autologging
        if "LGB" in mlflow_model_name:
            mlflow.lightgbm.autolog()
        elif "RF" in mlflow_model_name:
            mlflow.sklearn.autolog()

        # Log F1 score
        with mlflow.start_run(run_id=mlflow_model_run_id, nested=True):
            mlflow.log_metric(f"outsample_ovr_f1_{average}", f1_ovr)
            mlflow.log_metric(f"outsample_class0v1_f1", f1_ovo_01)
            mlflow.log_metric(f"outsample_class0v2_f1", f1_ovo_02)
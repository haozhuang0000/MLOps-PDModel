from src.mlops.abstractions import EvaluationABC
from src.mlops.model import ModelLoader
from typing import Literal
import numpy as np
import pandas as pd
import os
import mlflow
from sklearn.metrics import roc_auc_score

class ArCredit(EvaluationABC):

    def evaluate(self,
                 y_pred: np.array,
                 y_proba: np.array,
                 y_test: pd.Series,
                 mlflow_model_name: str,
                 mlflow_model_run_id: str,
                 average: str = 'macro',
                 multi_class: str = 'ovo') -> float:

        # model, model_uri = ModelLoader.load_model(mlflow_model_name, mlflow_model_run_id)
        #
        # # Predict probabilities
        # if method == 'classifier':
        #     y_proba = model.predict_proba(X_test)
        # else:
        #     y_proba = model.predict(X_test)

        # df = pd.DataFrame({"y": y_test, "y_pred": y_proba[:, 1]})
        # df = df[df.y <= 1]
        # df = df[df.y_pred > 0]
        # auc = roc_auc_score(df['y'], df['y_pred'])

        # Binarize the true labels: 1 if class is 1, else 0
        y_binary = (y_test == 1).astype(int)

        # No filtering on y_pred
        auc = roc_auc_score(y_binary, y_proba[:, 1])
        accuracy_ratio_ovr = 2 * auc - 1

        # from src.mlops.evaluation import evaluate_outsample_ovo
        # metric = 'accuracy_ratio'
        # ar_ovo_01 = evaluate_outsample_ovo(metric, y_test, None, y_proba, class_a=0, class_b=1)
        # ar_ovo_02 = evaluate_outsample_ovo(metric, y_test, None, y_proba, class_a=0, class_b=2)

        # Enable autologging
        if "LGB" in mlflow_model_name:
            mlflow.lightgbm.autolog()
        elif "RF" in mlflow_model_name:
            mlflow.sklearn.autolog()

        # Log AUC
        with mlflow.start_run(run_id=mlflow_model_run_id, nested=True):
            mlflow.log_metric(f"outsample_class0v1_ar", accuracy_ratio_ovr)
            # mlflow.log_metric(f"outsample_class_0v1_ar", ar_ovo_01)
            # mlflow.log_metric(f"outsample_class_0v2_ar", ar_ovo_02)

        return accuracy_ratio_ovr
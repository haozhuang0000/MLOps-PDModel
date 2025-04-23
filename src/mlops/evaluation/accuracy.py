from src.mlops.abstractions import EvaluationABC
from src.mlops.model import ModelLoader
from typing import Literal
import numpy as np
import pandas as pd
import os
import mlflow
from sklearn.metrics import accuracy_score

class Accuracy(EvaluationABC):

    def evaluate(self,
                 y_pred: np.array,
                 y_proba: np.array,
                 y_test: pd.Series,
                 mlflow_model_name: str,
                 mlflow_model_run_id: str,
                 average: None) -> float:

        # model, model_uri = ModelLoader.load_model(mlflow_model_name, mlflow_model_run_id)

        # # Predict class probabilities or labels
        # if method == 'classifier':
        #     y_proba = model.predict_proba(X_test)
        # else:
        #     y_proba = model.predict(X_test)
        #
        # # If the model outputs probabilities, use argmax to get class labels
        # if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
        #     y_pred = np.argmax(y_proba, axis=1)

        accuracy = accuracy_score(y_test, y_pred)

        # Enable autologging
        if "LGB" in mlflow_model_name:
            mlflow.lightgbm.autolog()
        elif "RF" in mlflow_model_name:
            mlflow.sklearn.autolog()

        # Log metric under current or nested run
        with mlflow.start_run(run_id=mlflow_model_run_id, nested=True):
            mlflow.log_metric("outsample_accuracy", accuracy)

        return accuracy
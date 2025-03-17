from src.mlops.abstractions import EvaluationABC
from src.mlops.model import ModelLoader
import numpy as np
import pandas as pd
import os
import mlflow
from sklearn.metrics import mean_squared_error, r2_score

class MSE(EvaluationABC):

    def evaluate(self,
                 df_X_test: pd.DataFrame,
                 df_y_true: pd.Series,
                 mlflow_model_name: str,
                 mlflow_model_run_id: str) -> float:

        model, model_uri = ModelLoader.load_model(mlflow_model_name, mlflow_model_run_id)
        y_pred = model.predict(df_X_test)
        mse = mean_squared_error(df_y_true, y_pred)
        # Enable autologging
        if "LGB" in mlflow_model_name:
            mlflow.lightgbm.autolog()
        elif "RF" in mlflow_model_name:
            mlflow.sklearn.autolog()

        with mlflow.start_run(run_id=mlflow_model_run_id, nested=True):
            mlflow.log_metric("mse", mse)

        # mlflow.register_model(model_uri=model_uri, name=mlflow_model_name)
        # print(f"Model registered as {registered_model_name}")
        return mse
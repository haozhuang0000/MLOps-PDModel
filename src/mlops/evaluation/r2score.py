from src.mlops.abstractions import EvaluationABC
from src.mlops.model import ModelLoader
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import mlflow

class R2Score(EvaluationABC):

    def evaluate(self,
                 df_X_test: pd.DataFrame,
                 df_y_true: pd.Series,
                 mlflow_model_name: str,
                 mlflow_model_run_id: str) -> float:

        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            mse: float
        """
        model, model_uri = ModelLoader.load_model(mlflow_model_name, mlflow_model_run_id)
        y_pred = model.predict(df_X_test)
        r2 = r2_score(df_y_true, y_pred)
        # Enable autologging
        if "LGB" in mlflow_model_name:
            mlflow.lightgbm.autolog()
        elif "RF" in mlflow_model_name:
            mlflow.sklearn.autolog()

        with mlflow.start_run(run_id=mlflow_model_run_id, nested=True):
            mlflow.log_metric("r2", r2)

        mlflow.register_model(model_uri=model_uri, name=mlflow_model_name)
        # mlflow.end_run()
        # print(f"Model registered as {registered_model_name}")
        return r2
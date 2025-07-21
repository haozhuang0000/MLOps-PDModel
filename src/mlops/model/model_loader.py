import mlflow
from mlflow import MlflowClient
from typing import Union, Any
import os
# from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
# Get the tracking URI
# tracking_uri = get_tracking_uri()

class ModelLoader(object):

    @staticmethod
    def load_model(model_name, run_id):

        if 'LGB' in model_name:
            # Load the logged model using the run ID
            # model_uri = f"runs:/{run_id}/{model_name}"
            model_uri = f"runs:/{run_id}/model"
            model = mlflow.lightgbm.load_model(model_uri)
        elif 'RF' in model_name:
            # model_uri = f"runs:/{run_id}/{model_name}"
            model_uri = f"runs:/{run_id}/model"
            model = mlflow.sklearn.load_model(model_uri)

        return model, model_uri

    @staticmethod
    def load_registered_model(model_name) -> Union[Any]:
        tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:8885')
        mlflow.set_tracking_uri(tracking_uri)
        # client = MlflowClient()
        # get latest model
        # models = client.get_latest_versions(model_name, stages=["None"])
        model_version = "latest"
        model_uri = f"models:/{model_name}/{model_version}"
        model = mlflow.lightgbm.load_model(model_uri)
        return model
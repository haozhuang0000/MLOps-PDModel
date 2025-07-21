import mlflow
from mlflow import MlflowClient
from typing import Tuple, Union, Dict, Any
from typing_extensions import Annotated
import pandas as pd
import pickle
import os
# from zenml.config import ResourceSettings
# from zenml import step
# from zenml.client import Client
# experiment_tracker = Client().active_stack.experiment_tracker
# from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

# Get the tracking URI
# tracking_uri = get_tracking_uri()
# @step(experiment_tracker=experiment_tracker.name, enable_cache=False, settings={"resources": ResourceSettings(cpu_count=5, gpu_count=4, memory="24GB")})
def load_registered_model(model_name) -> Union[Any]:
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:8885')
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    # get latest model
    models = client.get_latest_versions(model_name, stages=["None"])
    model_version = models[0].version
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.lightgbm.load_model(model_uri)
    return model

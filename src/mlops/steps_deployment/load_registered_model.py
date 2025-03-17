import mlflow
from typing import Tuple, Union, Dict, Any
from typing_extensions import Annotated
import pandas as pd
import pickle
import os
from zenml import step
from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

# Get the tracking URI
tracking_uri = get_tracking_uri()

@step(experiment_tracker=experiment_tracker.name, enable_cache=False)
def load_registered_model(model_name, id_bb_unique, y, model_version="1") -> Union[Any]:

    model_registered_name = model_name + '_' + id_bb_unique + '_' + y
    model_uri = f"models:/{model_registered_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    return model

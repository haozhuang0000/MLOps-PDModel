import json
import os
from pickle import FALSE
import numpy as np
import pandas as pd
# from materializer.custom_materializer import cs_materializer
from src.mlops.steps_deployment import (
    load_data,
    load_registered_model,
    predict
)
# from zenml import pipeline, step
# from zenml.config import DockerSettings
# from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
# from zenml.integrations.constants import MLFLOW, TENSORFLOW
# from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
#     MLFlowModelDeployer,
# )
# from zenml.integrations.mlflow.services import MLFlowDeploymentService
# from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
# from zenml.steps import Output

# docker_settings = DockerSettings(required_integrations=[MLFLOW])
import pandas as pd


# @pipeline(enable_cache=False)
def prediction_service(
        model_name:str,
        id_bb_unique: str,
        y: str,
        year: int
):
    X = load_data(id_bb_unique, y, year)
    model = load_registered_model(model_name, id_bb_unique, y)
    prediction = predict(model, X)
    print(X)
    print(prediction)
    return prediction

if __name__ == "__main__":
    prediction_service('LGBRegression', 'EQ0000000000142041', 'EBITDA', 2027)
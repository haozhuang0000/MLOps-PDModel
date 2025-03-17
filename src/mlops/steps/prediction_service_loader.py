# import json
#
# # from .utils import get_data_for_test
# import os
# from pickle import FALSE
#
# from scripts.steps.config import ModelNameConfig
# import numpy as np
# import pandas as pd
# # from materializer.custom_materializer import cs_materializer
# from src.mlops.steps import (
#     load_data,
#     load_intermediate_data,
#     clean_data,
#     split_data,
#     impute_data,
#     partition_data,
#     reconstruct_data,
#     train_model,
#     evaluate_model,
# )
# from zenml import pipeline, step
# from zenml.config import DockerSettings
# from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
# from zenml.integrations.constants import MLFLOW, TENSORFLOW
# from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
#     MLFlowModelDeployer,
# )
# from zenml.integrations.mlflow.services import MLFlowDeploymentService
# from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
# # from zenml.steps import Output
#
# from .utils import get_data_for_test
#
# docker_settings = DockerSettings(required_integrations=[MLFLOW])
# import pandas as pd
#
#
# @step(enable_cache=False)
# def prediction_service_loader(
#     model_name: str,
#     company_id: str,
#     year: int,
#     y: str
# ):
#
#     model, model_uri = ModelLoader.load_model(mlflow_model_name, mlflow_model_run_id)
#
#     pass
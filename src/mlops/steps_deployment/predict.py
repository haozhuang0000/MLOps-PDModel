from mlflow.server.auth import alert

from src.mlops.logger.utils.logger import Log
from src.mlops.model import ModelLoader
from src.mlops.configs import Variables
from typing import Tuple, Union, Dict, Any, List
from typing_extensions import Annotated
import pandas as pd
import numpy as np
import os
# from zenml.config import ResourceSettings
# from zenml import step
# from zenml.client import Client
# experiment_tracker = Client().active_stack.experiment_tracker

# @step(experiment_tracker=experiment_tracker.name, enable_cache=False, settings={"resources": ResourceSettings(cpu_count=5, gpu_count=4, memory="24GB")})
def predict(model_name: str, X: pd.DataFrame, datadate: str) -> Tuple[pd.DataFrame, str]:
    logger = Log(f"{os.path.basename(__file__)}").getlog()
    try:
        model = ModelLoader.load_registered_model(model_name)
        features = Variables.x_cols_to_process_cn
        X_in = X[features]
        prediction = model.predict_proba(X_in)
        pd = prediction[:, 1]
        poe = prediction[:, 2]

        X["pd_1"] = pd
        X["poe_1"] = poe
        X['YYYYMMDD'] = datadate
        alert_signal = "Success"
    except:
        alert_signal = "Fail"
        logger.error(f"[predict] Failed to load model: {model_name}")
        return pd.DataFrame(), alert_signal
    X = X[['Comp_No', 'YYYY', 'MM', 'YYYYMMDD', 'pd_1', 'poe_1']]
    return X, alert_signal
from src.mlops.data_loading.get_daily_XnY import get_XY
from src.mlops.configs import Variables
from typing import Tuple, Union, Dict, Any
from typing_extensions import Annotated, Tuple
import pandas as pd
from datetime import datetime
import pickle
import os
from dotenv import load_dotenv
load_dotenv()
from zenml.config import ResourceSettings
from zenml import step
from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name, enable_cache=False, settings={"resources": ResourceSettings(cpu_count=5, gpu_count=4, memory="24GB")})
def load_data(x_path, y_path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X, y = get_XY(x_path, y_path)
    return X, y

if __name__ == "__main__":
    load_data(2)
from src.mlops.data_loading.get_daily_XnY import get_XY
from src.mlops.logger.utils.logger import Log
from src.mlops.configs import Variables
from typing import Tuple, Union, Dict, Any
from typing_extensions import Annotated, Tuple
import pandas as pd
from datetime import datetime
import pickle
import os
from dotenv import load_dotenv
load_dotenv()
import time
import smbclient
from zenml.config import ResourceSettings
from zenml import step
from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name, enable_cache=False, settings={"resources": ResourceSettings(cpu_count=5, gpu_count=4, memory="24GB")})
def load_data(x_path: str,
              y_path: str,
              timeout_sec: int = 21600,
              poll_interval_sec: int = 600) -> Tuple[pd.DataFrame, pd.DataFrame, str]:

    logger = Log(f"{os.path.basename(__file__)}").getlog()
    smbclient.register_session(
        os.getenv("FILEIP"),
        username=os.getenv("FILEIP_USERNAME"),
        password=os.getenv("FILEIP_PASSWORD")
    )
    for path in [x_path, y_path]:
        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            if smbclient.path.exists(path):
                logger.info(f"✅ Found file: {path}")
                break
            logger.info(f"⏳ Waiting for file: {path}")
            time.sleep(poll_interval_sec)
        else:
            alert_signal = "Fail"
            logger.info(f"⛔ Timeout: File not found - {path}")
            return pd.DataFrame(), pd.DataFrame(), alert_signal

    alert_signal = 'Success'
    X, y = get_XY(x_path, y_path)
    return X, y, alert_signal

if __name__ == "__main__":
    load_data(2)
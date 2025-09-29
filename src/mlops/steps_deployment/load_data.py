from src.mlops.data_loading.get_daily_XnY import get_XY, get_cripred
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
# from zenml.config import ResourceSettings
# from zenml import step
# from zenml.client import Client
# experiment_tracker = Client().active_stack.experiment_tracker

# @step(experiment_tracker=experiment_tracker.name, enable_cache=False, settings={"resources": ResourceSettings(cpu_count=5, gpu_count=4, memory="24GB")})
def load_data(x_path: str,
              y_path: str,
              cripd_path: str,
              cripoe_path: str,
              timeout_sec: int = 21600,
              poll_interval_sec: int = 600) -> Tuple[pd.DataFrame, pd.DataFrame, str]:

    logger = Log(f"{os.path.basename(__file__)}").getlog()
    smbclient.register_session(
        os.getenv("FILEIP"),
        username=os.getenv("FILEIP_USERNAME"),
        password=os.getenv("FILEIP_PASSWORD")
    )
    for path in [x_path, y_path, cripd_path, cripoe_path]:
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
    X = X.dropna(subset=['YYYY', 'MM'])
    X['year_month'] = pd.to_datetime(X['YYYY'].astype(int).astype(str) + '-' + X['MM'].astype(int).astype(str).str.zfill(2), format='%Y-%m')
    # Find the latest row per Comp_No
    latest_idx = X.groupby('Comp_No')['year_month'].idxmax()
    # Filter the DataFrame to only those rows
    output_X = X.loc[latest_idx].reset_index(drop=True).drop(columns=['year_month'])
    cripred = get_cripred(cripd_path, cripoe_path)
    y.yyyymmdd = y.yyyymmdd.astype(int)

    return output_X, y, cripred, alert_signal

if __name__ == "__main__":
    load_data(2)
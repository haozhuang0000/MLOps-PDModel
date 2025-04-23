from src.mlops.logger.utils.logger import Log
from typing import Tuple
import time
import smbclient
import scipy.io
import io
import pandas as pd
import os
import math
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
from zenml.config import ResourceSettings
from zenml import step
from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker

@step(enable_cache=False, experiment_tracker=experiment_tracker.name, settings={"resources": ResourceSettings(cpu_count=5, gpu_count=4, memory="24GB")})
def wait_for_files(econ: int,
                   timeout_sec: int = 21600,
                   poll_interval_sec: int = 600) -> Tuple[str, str, str]:
    """
    Waits for both data (.mat) files to be available via SMB. Returns file paths.
    """
    logger = Log(f"{os.path.basename(__file__)}").getlog()
    dt = datetime.today()
    date = int(str(dt.date()).replace('-', '')) - 1
    print(date)
    # date = 20250414

    smbclient.register_session(
        os.getenv("FILEIP"),
        username=os.getenv("FILEIP_USERNAME"),
        password=os.getenv("FILEIP_PASSWORD")
    )
    base_dir = fr"\\{os.getenv('FILEIP')}\cri3\OfficialTest_AggDTD_SBChinaNA\ProductionData\Recent\Daily"
    folder_name = None
    # Wait for folder
    start_time = time.time()
    while time.time() - start_time < timeout_sec:
        try:
            all_files = smbclient.listdir(base_dir)
            matching = [f for f in all_files if f.startswith(f"{date}_cali_")]
            if matching:
                folder_name = matching[0]
                logger.info(f"✅ Found folder: {folder_name}")
                break
        except Exception as e:
            logger.info(f"⚠️ SMB access error: {e}")
        time.sleep(poll_interval_sec)

    if not folder_name:
        alert_signal = "Fail"
        logger.error(f"⛔ Timeout: Folder for date {date} not found in {base_dir}")
        return 'Fail', 'Fail', alert_signal
    else:
        alert_signal = "Success"

    folder_path = os.path.join(base_dir, folder_name)
    x_path = os.path.join(folder_path, r"Processing\P2_Pd\covariates\final", f"DATA_{econ}.mat")
    y_path = os.path.join(folder_path, r"IDMTData\Smart\FirmHistory", f"FirmHistory_{econ}.mat")

    return x_path, y_path, alert_signal
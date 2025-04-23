from src.mlops.logger import LoggerDescriptor
import pandas as pd
from typing import Union
import smbclient
import pandas as pd
import os

class DataLoader(object):
    @staticmethod
    def load_data(data_path: str) -> pd.DataFrame:
        df = pd.read_csv(data_path)
        return df

    @staticmethod
    def load_windows_data(data_path: str) -> pd.DataFrame:
        smbclient.register_session(
            os.getenv("FILEIP"),
            username=os.getenv("FILEIP_USERNAME"),
            password=os.getenv("FILEIP_PASSWORD")
        )

        if data_path.endswith('.csv'):
            with smbclient.open_file(data_path, mode='rb') as f:
                df = pd.read_csv(f)
            return df
        else:
            raise ValueError("Only .csv files are supported")


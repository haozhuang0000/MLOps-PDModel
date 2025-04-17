from src.mlops.logger import LoggerDescriptor
import pandas as pd
from typing import Union

class DataLoader:

    logger = LoggerDescriptor()
    def load_data(self,
                  data_path: str) -> pd.DataFrame:

        self.logger.info('loading x and y data...')

        df = pd.read_csv(data_path)


        return df
from src.mlops.logger.utils.logger import Log
import pandas as pd
from datetime import datetime
import os
from sqlalchemy.exc import ProgrammingError
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
load_dotenv()
from zenml.config import ResourceSettings
from zenml import step
from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker

@step(enable_cache=False, experiment_tracker=experiment_tracker.name, settings={"resources": ResourceSettings(cpu_count=5, gpu_count=4, memory="24GB")})
def save_mysql(df: pd.DataFrame, econ: int, task_date: str) -> str:
    try:
        df.dropna(subset=['Comp_No', 'YYYY', 'MM'], inplace=True)
        logger = Log(f"{os.path.basename(__file__)}").getlog()
        table_name = 'pd_daily'
        if task_date is None:
            task_date = datetime.today().strftime('%Y-%m-%d')

        # MySQL connection URL: dialect+driver://username:password@host/database
        engine = create_engine(f'mysql+pymysql://{os.getenv("MYSQL_USER")}:{os.getenv("MYSQL_PASS")}@{os.getenv("MYSQL_HOST")}/mlops_pd')

        try:
            with engine.connect() as conn:
                result = conn.execute(
                    text(f"SELECT MAX(version) FROM {table_name} WHERE task_date = :task_date"),
                    {"task_date": task_date}
                ).scalar()
                next_version = (result or 0) + 1
        except ProgrammingError:
            logger.warning(f"Table `{table_name}` does not exist. Defaulting version to 1.")
            next_version = 1

        # Add metadata columns
        df["econ"] = econ
        df["task_date"] = task_date
        df["version"] = next_version

        df.to_sql(table_name, con=engine, index=False, if_exists="append")
        logger.info(f"Saved **{df.shape[0]}** rows to `{table_name}` for task_date={task_date}, version={next_version}")

        alert_signal = "Success"
        return alert_signal
    except:
        alert_signal = "Fail"
        logger.error(f"Failed to save data to MySQL: {os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASS')}@{os.getenv('MYSQL_HOST')}/mlops_pd")
        return alert_signal



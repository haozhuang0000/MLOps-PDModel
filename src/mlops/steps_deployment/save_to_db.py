from src.mlops.logger.utils.logger import Log
import pandas as pd
from datetime import datetime
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
load_dotenv()
from zenml.config import ResourceSettings
from zenml import step
from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker

@step(enable_cache=False, experiment_tracker=experiment_tracker.name, settings={"resources": ResourceSettings(cpu_count=5, gpu_count=4, memory="24GB")})
def save_mysql(df, econ: int, task_date: str=None):
    logger = Log(f"{os.path.basename(__file__)}").getlog()
    table_name = 'pd_daily'
    task_date = task_date or datetime.today().strftime("%Y-%m-%d")

    # MySQL connection URL: dialect+driver://username:password@host/database
    engine = create_engine(f'mysql+pymysql://{os.getenv("MYSQL_USER")}:{os.getenv("MYSQL_PASS")}@{os.getenv("CPU05_HOST")}/mlops_pd')

    # Determine latest version for the task_date
    with engine.connect() as conn:
        result = conn.execute(
            text(f"SELECT MAX(version) FROM {table_name} WHERE task_date = :task_date"),
            {"task_date": task_date}
        ).scalar()
        next_version = (result or 0) + 1

    # Add metadata columns
    df["econ"] = econ
    df["task_date"] = task_date
    df["version"] = next_version

    df.to_sql(table_name, con=engine, index=False, if_exists="append")
    logger.info(f"Saved **{df.shape[0]}** rows to `{table_name}` for task_date={task_date}, version={next_version}")



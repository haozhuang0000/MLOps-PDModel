from src.mlops.logger.utils.logger import Log
import pandas as pd
from datetime import datetime
import os
from sqlalchemy.exc import ProgrammingError
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
load_dotenv()
# from zenml.config import ResourceSettings
# from zenml import step
# from zenml.client import Client
# experiment_tracker = Client().active_stack.experiment_tracker

# @step(enable_cache=False, experiment_tracker=experiment_tracker.name, settings={"resources": ResourceSettings(cpu_count=5, gpu_count=4, memory="24GB")})
def save_mysql(df: pd.DataFrame,
               y: pd.DataFrame,
               cripred: pd.DataFrame,
               econ: int,
               task_date: str) -> str:
    try:
        df.dropna(subset=['comp_id', 'yyyy', 'mm'], inplace=True)
        logger = Log(f"{os.path.basename(__file__)}").getlog()
        table_name = 'mlpd_daily_dev'
        y_table_name = 'pd_ground_truth'
        cripred_table_name = 'cripd_daily'
        if task_date is None:
            task_date = datetime.today().strftime('%Y-%m-%d')
        else:
            task_date = datetime.strptime(task_date, '%Y%m%d')

        # MySQL connection URL: dialect+driver://username:password@host/database
        engine = create_engine(f'mysql+pymysql://{os.getenv("MYSQL_USER")}:{os.getenv("MYSQL_PASS")}@{os.getenv("MYSQL_HOST")}/mlops_pd')

        ## ------------------------------------- Prediction Table -------------------------------------
        df["econ"] = econ
        df["task_date"] = task_date
        # df["version"] = next_version
        df['yyyymmdd'] = pd.to_datetime(df['yyyymmdd'], format='%Y%m%d').dt.date
        df['task_date'] = pd.to_datetime(df['task_date'], format='%Y-%m-%d').dt.date
        try:
            with engine.connect() as conn:
                result = conn.execute(
                    text(f"SELECT * FROM {table_name} WHERE task_date = :task_date"),
                    {"task_date": task_date}
                )
                existing_row = result.fetchone()
                if existing_row:
                    logger.info(f"Data for task_date `{task_date}` already exists in `{table_name}`. Skipping insert.")
                else:
                    df.to_sql(table_name, con=engine, index=False, if_exists="append")
                    logger.info(f"Saved **{df.shape[0]}** rows to `{table_name}` for task_date={task_date}")
        except ProgrammingError:
            logger.warning(f"Table `{table_name}` does not exist. Defaulting version to 1.")

        ## ------------------------------------- Ground Truth Table -------------------------------------
        y['task_date'] = task_date
        y['yyyymmdd'] = pd.to_datetime(y['yyyymmdd'], format='%Y%m%d').dt.date
        y['task_date'] = pd.to_datetime(y['task_date'], format='%Y-%m-%d').dt.date
        try:
            with engine.connect() as conn:
                result = conn.execute(
                    text(f"SELECT * FROM {y_table_name} WHERE task_date = :task_date"),
                    {"task_date": task_date}
                )
                existing_row = result.fetchone()
            if existing_row:
                logger.info(f"Data for task_date `{task_date}` already exists in `{y_table_name}`. Skipping insert.")
            else:
                y.to_sql(y_table_name, con=engine, index=False, if_exists="append")
        except ProgrammingError:
            y.to_sql(y_table_name, con=engine, index=False, if_exists="append")
            logger.warning(f"Table `{y_table_name}` does not exist. Creating...")

        ## ------------------------------------- CRI PD Table -------------------------------------
        cripred['task_date'] = task_date
        cripred['task_date'] = pd.to_datetime(cripred['task_date'], format='%Y-%m-%d').dt.date
        try:
            with engine.connect() as conn:
                result = conn.execute(
                    text(f"SELECT * FROM {cripred_table_name} WHERE task_date = :task_date"),
                    {"task_date": task_date}
                )
                existing_row = result.fetchone()
            # existing_row = None
            if existing_row:
                logger.info(f"Data for task_date `{task_date}` already exists in `{cripred_table_name}`. Skipping insert.")
            else:
                cripred.to_sql(cripred_table_name, con=engine, index=False, if_exists="append")
        except ProgrammingError:
            cripred.to_sql(cripred_table_name, con=engine, index=False, if_exists="append")
            logger.warning(f"Table `{cripred_table_name}` does not exist. Creating...")

        alert_signal = "Success"
        return alert_signal
    except Exception as e:
        alert_signal = "Fail"
        logger.error(f"Failed to save data to MySQL: {os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASS')}@{os.getenv('MYSQL_HOST')}/mlops_pd")
        logger.error(e)
        return alert_signal



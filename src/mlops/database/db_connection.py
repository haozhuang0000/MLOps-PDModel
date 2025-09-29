from src.mlops.configs.db_config import MysqlConfig
import pandas as pd
from sqlalchemy import create_engine
import pymysql

class MySQLConnection:

    MYSQL_USER = MysqlConfig.MYSQL_USERNAME
    MYSQL_PASSWORD = MysqlConfig.MYSQL_PASSWORD
    MYSQL_HOST = MysqlConfig.MYSQL_HOST  # or your remote host/IP
    MYSQL_PORT = MysqlConfig.MYSQL_PORT

    def _get_engine(self, MYSQL_DATABASE='mlops_pd'):
        engine = create_engine(
            f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{MYSQL_DATABASE}"
        )
        return engine

    def _get_connection(self, MYSQL_DATABASE='mlops_pd'):

        return pymysql.connect(
            host=self.MYSQL_HOST,
            user=self.MYSQL_USER,
            password=self.MYSQL_PASSWORD,
            db=MYSQL_DATABASE,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True
        )
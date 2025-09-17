from src.mlops.database.db_connection import MySQLConnection
import pandas as pd


class MySQLDatabase:

    def __init__(self):

        self.connection = MySQLConnection()
        self.engine = self.connection._get_engine()

    def get_econ_table(self):
        df = pd.read_sql("select * from econ", self.engine)
        return df

    def get_metric_table(self):
        df = pd.read_sql("select * from metric", self.engine)
        return df

    def get_company_table(self):
        df = pd.read_sql("select * from company", self.engine)
        return df

    def get_metrics_type_table(self):
        df = pd.read_sql("select * from metrics_type", self.engine)
        return df

    def get_model_type_table(self):
        df = pd.read_sql("select * from model_type", self.engine)
        return df

    def get_target_table(self):
        df = pd.read_sql("select * from target", self.engine)
        return df
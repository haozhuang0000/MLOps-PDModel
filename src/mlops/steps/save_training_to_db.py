from src.mlops.logger.utils.logger import Log
from src.mlops.database.db_connection import MySQLConnection
from src.mlops.database.mysql_db import MySQLDatabase
import pandas as pd
from datetime import datetime
import os
from sqlalchemy.exc import ProgrammingError
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
load_dotenv()

class SaveToMySQL:

    def __init__(self):
        connection = MySQLConnection()
        self.conn = connection._get_connection()
        self.engine = connection._get_engine()

    def save_model_info_to_db(self, model_type_id, version, econ_id, target_id, train_date):
        with self.conn.cursor() as cursor:
            sql_insert = """
            INSERT IGNORE INTO model (model_type_id, version, econ_id, target_id, train_date)
            VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(sql_insert, (model_type_id, version, econ_id, target_id, train_date))

            sql_select = """
            SELECT model_id FROM model
            WHERE model_type_id = %s AND version = %s AND econ_id = %s AND target_id = %s AND train_date = %s
            """
            cursor.execute(sql_select, (model_type_id, version, econ_id, target_id, train_date))
            model_id = cursor.fetchone()[0]
        self.conn.close()
        return model_id

    def save_metrics_type_to_db(self, metrics_type_name):
        with self.conn.cursor() as cursor:
            cursor.execute(
                "INSERT IGNORE INTO metrics_type (metrics_name) VALUES (%s)",
                (metrics_type_name,)
            )
            cursor.execute(
                "SELECT metrics_type_id FROM metrics_type WHERE metrics_name = %s",
                (metrics_type_name,)
            )
            metrics_type_id = cursor.fetchone()[0]
        self.conn.close()
        return metrics_type_id

    def save_model_type_to_db(self, model_type_name):
        with self.conn.cursor() as cursor:
            cursor.execute(
                "INSERT IGNORE INTO model_type (model_name) VALUES (%s)",
                (model_type_name,)
            )
            cursor.execute(
                "SELECT model_type_id FROM model_type WHERE model_name = %s",
                (model_type_name,)
            )
            model_type_id = cursor.fetchone()[0]
        self.conn.close()
        return model_type_id

    def save_target_to_db(self, target_name):
        with self.conn.cursor() as cursor:
            cursor.execute(
                "INSERT IGNORE INTO target (target_name) VALUES (%s)",
                (target_name,)
            )
            cursor.execute(
                "SELECT target_id FROM target WHERE target_name = %s",
                (target_name,)
            )
            target_id = cursor.fetchone()[0]
        self.conn.close()
        return target_id

    def save_metrics_to_db(self, metrics_type_id, model_id, value):

        with self.conn.cursor() as cursor:
            sql = """
            INSERT INTO metrics (metrics_type_id, model_id, value)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE value = VALUES(value)
            """
            cursor.execute(sql, (metrics_type_id, model_id, value))
        self.conn.close()

    def save_mlpd_daily_df(self, df):
        """
        Save a pandas DataFrame to the mlpd_daily table.
        Expects columns: ['comp_no', 'yyyymmdd', 'model_id', 'target_id', 'econ_id', 'value', 'operation_date']
        """
        with self.conn.cursor() as cursor:
            sql = """
            INSERT INTO mlpd_daily (comp_no, yyyymmdd, model_id, target_id, econ_id, value, operation_date)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE value = VALUES(value), operation_date = VALUES(operation_date)
            """

            values = [
                (
                    row['comp_no'],
                    row['yyyymmdd'],
                    row['model_id'],
                    row['target_id'],
                    row['econ_id'],
                    row['value'],
                    row['operation_date']
                )
                for _, row in df.iterrows()
            ]

            cursor.executemany(sql, values)

        self.conn.close()

def save_training_info_into_mysql(model_info_dict, econ_id, target):

    mysql_saver = SaveToMySQL()
    # mysql_loader = MySQLDatabase()
    target_name = f'pd_poe_{target}'
    ## Save target
    target_id = mysql_saver.save_target_to_db(target_name)

    for k, v in model_info_dict.items():
        model_type_name = k
        model_version = v['version']
        metric_dict = v['metrics']
        train_date = v['train_date']

        ## Save model type name
        model_type_id = mysql_saver.save_model_type_to_db(model_type_name)

        ## Save model
        model_id = mysql_saver.save_model_info_to_db(model_type_id=model_type_id,
                                          version=model_version,
                                          econ_id=econ_id,
                                          target_id=target_id,
                                          train_date=train_date)

        ## Load model

        for key, value in metric_dict.items():
            ## Save metrics type name
            metrics_type_id = mysql_saver.save_metrics_type_to_db(key)

            ## Save metrics value
            mysql_saver.save_metrics_to_db(metrics_type_id=metrics_type_id, model_id=model_id, value=value)









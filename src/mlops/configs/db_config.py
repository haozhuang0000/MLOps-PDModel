import os

from dotenv import load_dotenv
load_dotenv()

class MysqlConfig:

    MYSQL_USER = os.getenv('MYSQL_USER')
    MYSQL_PASS = os.getenv('MYSQL_PASS')
    MYSQL_HOST = os.getenv('MYSQL_HOST')
    MYSQL_PORT = os.getenv('MYSQL_PORT')
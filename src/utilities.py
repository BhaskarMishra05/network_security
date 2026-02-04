import os
import sys
import pandas
import numpy
import mysql.connector
from sqlalchemy import create_engine
import yaml

def database_engine_loader():
    with open ('./credientials/database.yaml','r') as yaml_file_obj:
        config = yaml.safe_load(yaml_file_obj)

    db_config = config['database']

    DB_URL = (
        f'mysql+mysqlconnector://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['name']}')
    engine = create_engine(DB_URL)
    return engine


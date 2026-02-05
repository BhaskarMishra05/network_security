import pandas as pd
import numpy
import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utilities import database_engine_loader

def data_imputation_database(df, table_name, engine):
    try:
        logging.info(f"Starting data imputation for table {table_name}")
        df.to_sql(table_name, engine, if_exists = 'replace', index = False, chunksize = 7000)
        logging.info(f"Completed data imputation for table {table_name}")
    except Exception as e:
        raise CustomException(e,sys)

if __name__ == "__main__":
    
    try:
        for file in os.listdir('./Data_Housing/Database_inhibitor/'):
            if '.csv' in file:
                logging.info(f"Reading file {file}")
                df = pd.read_csv('./Data_Housing/Database_inhibitor/'+file)
                logging.info(f"File {file} loaded successfully")
                data_imputation_database(df, table_name = file[:-4], engine = database_engine_loader())
    except Exception as e:
        raise CustomException(e,sys)
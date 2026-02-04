import pandas as pd
import numpy
import os
import sys
from src.utilities import database_engine_loader

def data_imputation_database(df, table_name, engine):
    df.to_sql(table_name, engine, if_exists = 'replace', index = False, chunksize = 7000)
for file in os.listdir('./Data_Housing/Database_inhibitor/'):
    if '.csv' in file:
        df = pd.read_csv('./Data_Housing/Database_inhibitor/'+file)
        data_imputation_database(df, table_name = file[:-4], engine = database_engine_loader())

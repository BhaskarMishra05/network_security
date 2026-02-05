from src.logger import logging
from src.exception import CustomException
import os
import sys
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from src.utilities import database_engine_loader
from src.database_inhibitor import data_imputation_database


class DATA_INGESTION():

    def data_ingestion_method(self, relation_to_fetch: str):
        try:
            logging.info("Starting data ingestion method")
            engine = database_engine_loader()
            logging.info("Database engine loaded")
            query = f'SELECT * FROM {relation_to_fetch}'
            logging.info(f"Executing query on relation {relation_to_fetch}")
            relation_dataframe = pd.read_sql(query, engine)
            logging.info("Data fetched into dataframe")
            logging.info(f"Shape of dataframe: {relation_dataframe.shape}")


            relation_train, relation_test = train_test_split(relation_dataframe, test_size=0.3, random_state=42)
            logging.info("Train test split completed")
            
            logging.info("Starting training data imputation")
            data_imputation_database(relation_train,table_name = 'training_dataset', engine= engine)
            logging.info("Training data imputation completed")

            logging.info("Starting testing data imputation")
            data_imputation_database(relation_test, table_name = 'testing_dataset', engine = engine)
            logging.info("Testing data imputation completed")
        except Exception as e:
            raise CustomException(e,sys)
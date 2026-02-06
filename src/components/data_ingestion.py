from src.logger import logging
from src.exception import CustomException
import os
import sys
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from src.utilities import database_engine_loader
from src.database_inhibitor import data_imputation_database
from dataclasses import dataclass

@dataclass
class DATA_INGESTION_CONFIG():
    training_dataset_path: str = os.path.join('Data_Housing/artifacts','training_dataset.csv')
    testing_dataset_path: str = os.path.join('Data_Housing/artifacts','testing_dataset.csv')

class DATA_INGESTION():
    def __init__(self):
        self.data_ingestion_config_object = DATA_INGESTION_CONFIG()

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
            relation_train.to_csv(self.data_ingestion_config_object.training_dataset_path, index = False)
            logging.info("Training data imputation completed")

            logging.info("Starting testing data imputation")
            relation_test.to_csv(self.data_ingestion_config_object.testing_dataset_path, index= False)
            logging.info("Testing data imputation completed")

            return (
                self.data_ingestion_config_object.testing_dataset_path,
                self.data_ingestion_config_object.training_dataset_path
            )
        except Exception as e:
            raise CustomException(e,sys)
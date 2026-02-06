from src.components.data_ingestion import DATA_INGESTION, DATA_INGESTION_CONFIG
from src.components.data_transformation import DATA_TRANSFORMATION
from src.components.model_trainer import MODEL_TRAINER
from src.exception import CustomException
from src.logger import logging
import os
import sys


'''try:
    data_ingestion_object = DATA_INGESTION()
    train_ , test_ = data_ingestion_object.data_ingestion_method('database_raw')
except Exception as e:
    raise CustomException (e,sys)'''

try: 
    data_ingestion_config = DATA_INGESTION_CONFIG()
    train_path = data_ingestion_config.training_dataset_path
    test_path = data_ingestion_config.testing_dataset_path 


except Exception as e:
    raise CustomException(e,sys)
try:
    data_transformation_object = DATA_TRANSFORMATION()
    train_arry, test_arry = data_transformation_object.transformation(train_path= train_path, test_path= test_path)
except Exception as e:
    raise CustomException (e,sys)

try:
    model_trainer_object = MODEL_TRAINER()
    model_trainer_object.training_method(train_arry, test_arry)
except Exception as e:
    raise CustomException(e,sys)
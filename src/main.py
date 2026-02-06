from src.components.data_ingestion import DATA_INGESTION
from src.exception import CustomException
from src.logger import logging
import os
import sys

try:
    data_ingestion_object = DATA_INGESTION()
    data_ingestion_object.data_ingestion_method('database_raw')
except Exception as e:
    raise CustomException (e,sys)
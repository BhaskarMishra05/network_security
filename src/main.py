from src.components.data_ingestion import DATA_INGESTION

try:
    data_ingestion_object = DATA_INGESTION()
    data_ingestion_object.data_ingestion_method('final_dataset_with_all_features')
except Exception as e:
    raise CustomException (e,sys)
import os
import sys
from sqlalchemy import create_engine
import yaml
import pickle

def database_engine_loader():
    with open ('/home/korty/Code_House/VS_CODE/Network Security/credientials/database.yaml','r') as yaml_file_obj:
        config = yaml.safe_load(yaml_file_obj)

    db_config = config['database']

    DB_URL = (
        f"mysql+mysqlconnector://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['name']}")
    engine = create_engine(DB_URL)
    print(engine)
    return engine

def load_object (file_path: str):
    with open (file_path, 'rb') as file_object:
        return pickle.load(file_object)
    
def save_object (file_path , object):
    os.makedirs(os.path.dirname(file_path), exist_ok= True)
    with open (file_path , 'wb') as file_object:
        return pickle.dump(object, file_object)
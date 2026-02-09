import os
import sys
import pandas as pd
from pandas import DataFrame
import numpy as np 
from src.logger import logging
from src.exception import CustomException
from src.utilities import load_object, save_object
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import SimpleImputer
from dataclasses import dataclass


@dataclass
class DATA_TRANSFORMATION_CONFIG:
    preprocessing_path: str = os.path.join('Data_Housing/model_training_files', 'preprocessing.pkl')

class DATA_TRANSFORMATION:
    def __init__(self):
        self.data_transformation_config = DATA_TRANSFORMATION_CONFIG()
        logging.info("DATA_TRANSFORMATION class initialized")

    def dataframe_clearning(self, df: DataFrame)-> DataFrame:
        try:
            logging.info("Starting dataframe_clearning method")

            top20_cols = ['abnormal_url','phish_long_path','phish_adv_number_count','https',
            'web_ssl_valid','suspicious_extension','phish_adv_many_subdomains','phish_adv_long_domain',
            'phish_urgency_words','web_security_score','url_len','%%',
            '+','phish_adv_hyphen_count','phish_multiple_subdomains','phish_adv_many_params',
            '?','web_xframe','=','phish_adv_exact_brand_match','label'
            ]

            df_new = df.loc[:,top20_cols]

            logging.info(f"dataframe_clearning completed. Shape: {df_new.shape}")

            return df_new
        except Exception as e:
            logging.info("Error occurred in dataframe_clearning")
            raise CustomException(e,sys)

    def preprocessing_method(self, df: DataFrame):
        try:
            logging.info("Starting preprocessing_method")

            numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            numerical_pipeline = Pipeline([('numerical_imputer', SimpleImputer(strategy= 'median')),
                                        ('scaler', StandardScaler())])
            
            categorical_pipeline = Pipeline([('categorical_imputer', SimpleImputer(strategy='most_frequent')),
                                            ('encoder', LabelEncoder())])

            preprocessing_CT = ColumnTransformer([('numeical_CT', numerical_pipeline, numerical_columns),
                                                ('categorical_CT', categorical_pipeline, categorical_columns)])
            
            logging.info("preprocessing_method completed successfully")

            return preprocessing_CT
        except Exception as e:
            logging.info("Error occurred in preprocessing_method")
            raise CustomException(e,sys)
        
    def transformation(self, train_path, test_path):
        try: 
            logging.info("Starting transformation method")

            train = pd.read_csv(train_path)
            test = pd.read_csv(test_path)

            logging.info(f"Train data loaded. Shape: {train.shape}")
            logging.info(f"Test data loaded. Shape: {test.shape}")

            target = 'label'

            train = self.dataframe_clearning(df= train)
            test = self.dataframe_clearning(df= test)

            features_train= train.drop(columns=[target])
            target_train= train[target]

            features_test = test.drop(columns=[target])
            target_test = test[target]

            logging.info(f"Features train shape: {features_train.shape}")
            logging.info(f"Target train distribution:\n{target_train.value_counts()}")

            random_sampler = RandomUnderSampler()

            resampled_features_train, resampled_target_train = random_sampler.fit_resample(features_train, target_train)

            logging.info("RandomUnderSampler applied")
            logging.info(f"Resampled features shape: {resampled_features_train.shape}")
            logging.info(f"Resampled target distribution:\n{resampled_target_train.value_counts()}")

            preprocessing_object = self.preprocessing_method(resampled_features_train)

            transformed_train = preprocessing_object.fit_transform(resampled_features_train)
            transformed_test = preprocessing_object.transform(features_test)

            logging.info(f"Transformed train shape: {transformed_train.shape}")
            logging.info(f"Transformed test shape: {transformed_test.shape}")

            train_array = np.c_[transformed_train, resampled_target_train]
            test_array = np.c_[transformed_test, target_test]

            logging.info("Combining features and target into final arrays")

            save_object(self.data_transformation_config.preprocessing_path, preprocessing_object)

            logging.info(f"Preprocessing object saved at: {self.data_transformation_config.preprocessing_path}")
            logging.info("transformation method completed successfully")

            return (
                train_array, 
                test_array
            )
        except Exception as e:
            logging.info("Error occurred in transformation method")
            raise CustomException(e,sys)

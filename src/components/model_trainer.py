import os
import sys
import joblib
import pandas as pd
import numpy as np
from dataclasses import dataclass
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              classification_report, recall_score, precision_score,
                              f1_score)
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from src.utilities import save_object
from src.logger import logging
from src.exception import CustomException

@dataclass
class MODEL_TRAINER_CONFIG:
    model_trainer_file_path = os.path.join('Data_Housing/model_training_files','model.pkl')

class MODEL_TRAINER:
    def __init__(self):
        self.model_trainer_config_object = MODEL_TRAINER_CONFIG()
        logging.info("MODEL_TRAINER class initialized")

    def training_method(self, train_array, test_array):
        try:
            logging.info("Starting training_method")

            X_train = train_array[:,:-1]
            y_train = train_array[:,-1]
            X_test = test_array[:,:-1]
            y_test = test_array[:,-1]

            logging.info(f"X_train shape: {X_train.shape}")
            logging.info(f"y_train shape: {y_train.shape}")
            logging.info(f"X_test shape: {X_test.shape}")
            logging.info(f"y_test shape: {y_test.shape}")

            xgbc = XGBClassifier()
            hgbc = HistGradientBoostingClassifier()
            gbc = GradientBoostingClassifier()
            lr = LogisticRegression()

            logging.info("Base models initialized")

            model_voting = VotingClassifier(
                estimators=[
                    ('xgbc', xgbc),
                    ('lr', lr)
                ],
                voting='soft'
            )

            logging.info("VotingClassifier created")

            model = Pipeline([
                ('sampler', RandomUnderSampler(random_state=42)),
                ('model', model_voting)
            ])

            logging.info("Pipeline created, starting model training")

            model.fit(X_train, y_train)

            logging.info("Model training completed")

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            confusionmatrix = confusion_matrix(y_test, y_pred)
            claasifictionreport = classification_report(y_test, y_pred)
            recall = recall_score(y_test, y_pred, average= 'weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average= 'weighted')

            logging.info(f"Accuracy: {accuracy}")
            logging.info(f"Precision: {precision}")
            logging.info(f"Recall: {recall}")
            logging.info(f"F1 Score: {f1}")
            logging.info(f"Confusion Matrix :{confusionmatrix}")
            logging.info(f"Classification Report: {claasifictionreport}")
            save_object(
                self.model_trainer_config_object.model_trainer_file_path,
                model
            )

            logging.info(f"Model saved at {self.model_trainer_config_object.model_trainer_file_path}")
            logging.info("training_method completed successfully")

            return (accuracy, precision, recall, f1)

        except Exception as e:
            logging.info("Error occurred in training_method")
            raise CustomException(e, sys)

import os
import sys
import joblib
import pandas as pd
import numpy as np
from dataclasses import dataclass
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.under_sampling import RandomUnderSampler


@dataclass
class MODEL_TRAINER_CONFIG:
    model_trainer_file_path = os.path.join('Data_Housing/model_training_files','model.pkl')

class MODEL_TRAINER:
    def __init__(self):
        self.model_trainer_config_object = MODEL_TRAINER_CONFIG()

    def training_method(self, train_array, test_array):
        X_train = train_array[:,:-1]
        y_train = train_array[:,-1]
        X_test = test_array[:,:-1]
        y_test = test_array[:,-1]

        model = XGBClassifier()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc_scr = accuracy_score(y_test, y_pred)
        conf_mat = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        print('Accu ---> {acc_scr}')
        print(conf_mat)
        print()
        print(class_report)
from src.utilities import load_object
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
from pandas import DataFrame

class PREDICTION_PIPELINE:
    def __init__(self):
        self.model_file_path = 'Data_Housing/model_training_files/model.pkl'
        self.preprocessed_file_path = 'Data_Housing/model_training_files/preprocessing.pkl'

    def predict(self, features: DataFrame):
        features = features.copy()
        model = load_object(file_path= self.model_file_path)
        preprocessor = load_object(file_path= self.preprocessed_file_path)

        transformed_data = preprocessor.transform(features)
        pred = model.predict(transformed_data)
        return pred
    
class CUSTOM_DATASET:
    def __init__(self,
                 abnormal_url,
                 phish_long_path,
                 phish_adv_number_count,
                 https,
                 web_ssl_valid,
                 suspicious_extension,
                 phish_adv_many_subdomains,
                 phish_adv_long_domain,
                 phish_urgency_words,
                 web_security_score,
                 url_len,
                 percent_count,
                 plus_count,
                 phish_adv_hyphen_count,
                 phish_multiple_subdomains,
                 phish_adv_many_params,
                 question_mark_count,
                 web_xframe,
                 equal_count,
                 phish_adv_exact_brand_match):

        self.abnormal_url = abnormal_url
        self.phish_long_path = phish_long_path
        self.phish_adv_number_count = phish_adv_number_count
        self.https = https
        self.web_ssl_valid = web_ssl_valid
        self.suspicious_extension = suspicious_extension
        self.phish_adv_many_subdomains = phish_adv_many_subdomains
        self.phish_adv_long_domain = phish_adv_long_domain
        self.phish_urgency_words = phish_urgency_words
        self.web_security_score = web_security_score
        self.url_len = url_len
        self.percent_count = percent_count
        self.plus_count = plus_count
        self.phish_adv_hyphen_count = phish_adv_hyphen_count
        self.phish_multiple_subdomains = phish_multiple_subdomains
        self.phish_adv_many_params = phish_adv_many_params
        self.question_mark_count = question_mark_count
        self.web_xframe = web_xframe
        self.equal_count = equal_count
        self.phish_adv_exact_brand_match = phish_adv_exact_brand_match
    def dataset(self):
        data =  {
            "abnormal_url": self.abnormal_url,
            "phish_long_path": self.phish_long_path,
            "phish_adv_number_count": self.phish_adv_number_count,
            "https": self.https,
            "web_ssl_valid": self.web_ssl_valid,
            "suspicious_extension": self.suspicious_extension,
            "phish_adv_many_subdomains": self.phish_adv_many_subdomains,
            "phish_adv_long_domain": self.phish_adv_long_domain,
            "phish_urgency_words": self.phish_urgency_words,
            "web_security_score": self.web_security_score,
            "url_len": self.url_len,
            "percent_count": self.percent_count,
            "plus_count": self.plus_count,
            "phish_adv_hyphen_count": self.phish_adv_hyphen_count,
            "phish_multiple_subdomains": self.phish_multiple_subdomains,
            "phish_adv_many_params": self.phish_adv_many_params,
            "question_mark_count": self.question_mark_count,
            "web_xframe": self.web_xframe,
            "equal_count": self.equal_count,
            "phish_adv_exact_brand_match": self.phish_adv_exact_brand_match
        }

        return data

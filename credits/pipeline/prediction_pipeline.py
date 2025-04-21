import os
import sys

import numpy as np
import pandas as pd
from credits.entity.config_entity import CredictorConfig


from credits.cloud_storage.aws_storage import SimpleStorageService
from credits.exception import creditsException
from credits.logger import logging
from credits.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from credits.entity.config_entity import ModelPusherConfig
from credits.entity.s3_estimator import  CredEstimator


from credits.utils.main_utils import read_yaml_file
from pandas import DataFrame

from credits.exception import creditsException
from credits.logger import logging


class CREDITData:
    def __init__(self,
                person_age,
                person_income,
                person_home_ownership,
                person_emp_length,
                loan_intent,
                loan_grade,
                loan_amnt,
                loan_int_rate,
                loan_percent_income,
                cb_person_default_on_file,
                cb_person_cred_hist_length
                ):
        """
        CREDIT Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.person_age = person_age
            self.person_income = person_income
            self.person_home_ownership = person_home_ownership
            self.person_emp_length = person_emp_length
            self.loan_intent = loan_intent
            self.loan_amnt = loan_amnt
            self.loan_percent_income = loan_percent_income
            self.loan_int_rate = loan_int_rate
            self.cb_person_default_on_file = cb_person_default_on_file
            self.cb_person_cred_hist_length = cb_person_cred_hist_length
            self.loan_grade = loan_grade


        except Exception as e:
            raise creditsException(e, sys) from e

    def get_credit_input_data_frame(self)-> DataFrame:
        """
        This function returns a DataFrame from CREDITData class input
        """
        try:
            
            credit_input_dict = self.get_credt_data_as_dict()
            return DataFrame(credit_input_dict)
        
        except Exception as e:
            raise creditsException(e, sys) from e


    def get_credt_data_as_dict(self):
        """
        This function returns a dictionary from creditData class input 
        """
        logging.info("Entered get_credt_data_as_dict method as creditData class")

        try:
            input_data = {
                "person_age": [self.person_age],
                "person_income": [self.person_income],
                "person_home_ownership": [self.person_home_ownership],
                "person_emp_length": [self.person_emp_length],
                "loan_intent": [self.loan_intent],
                "loan_amnt": [self.loan_amnt],
                "loan_grade": [self.loan_grade],
                "loan_int_rate": [self.loan_int_rate],
                "loan_percent_income": [self.loan_percent_income],
                "cb_person_default_on_file": [self.cb_person_default_on_file],
                "cb_person_cred_hist_length": [self.cb_person_cred_hist_length],
            }

            logging.info("Created credt data dict")

            logging.info("Exited get_credt_data_as_dict method as credtdata class")

            return input_data

        except Exception as e:
            raise creditsException(e, sys) from e

class CREDTClassifier:
    def __init__(self,prediction_pipeline_config: CredictorConfig = CredictorConfig(),) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            # self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise creditsException(e, sys)


    def predict(self, dataframe) -> str:
        """
        
        Returns: Prediction in string format
        """
        try:
            logging.info("Entered predict method of Classifier class")
            model =  CredEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result =  model.predict(dataframe)
            
            return result
        
        except Exception as e:
            raise creditsException(e, sys)
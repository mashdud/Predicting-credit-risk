import json
import sys

import pandas as pd
import sys

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from pandas import DataFrame

from  credits.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from credits.entity.config_entity import DataIngestionConfig
from credits.entity.artifact_entity import DataIngestionArtifact
from credits.exception import creditsException
from credits.logger import logging
from credits.data_access.credit_data import Creditdata


from  credits.utils.main_utils import read_yaml_file, write_yaml_file
from  credits.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact

from credits.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from credits.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file, drop_columns




import sys
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from imblearn.combine import SMOTEENN

from credits.exception import creditsException
from credits.logger import logging
from credits.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file, drop_columns
from credits.entity.config_entity import DataTransformationConfig
from credits.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from credits.entity.estimator import  CBPersonDefaultOnFile
from credits.constants import TARGET_COLUMN, SCHEMA_FILE_PATH

def drop_outliers_from_schema(X: pd.DataFrame, y: pd.Series, schema_config: dict):
    rules = schema_config.get("outlier_rules", [])
    X_cleaned = X.copy()
    y_cleaned = y.copy()
    for rule in rules:
        col = rule["column"]
        max_val = rule.get("max")
        if max_val is not None and col in X_cleaned.columns:
            mask = X_cleaned[col] <= max_val
            X_cleaned = X_cleaned[mask]
            y_cleaned = y_cleaned[mask]
    return X_cleaned, y_cleaned

def impute_missing_values_from_schema(X: pd.DataFrame, schema_config: dict) -> pd.DataFrame:
    impute_config = schema_config.get("imputation", {})
    columns = impute_config.get("columns", [])
    neighbors = impute_config.get("neighbors", 5)
    if not columns:
        return X
    imputer = KNNImputer(n_neighbors=neighbors)
    X_copy = X.copy()
    X_copy[columns] = imputer.fit_transform(X_copy[columns])
    return X_copy


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: configuration for data transformation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise creditsException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise creditsException(e, sys)



    def get_data_transformer_object(self) -> Pipeline:
    
        logging.info(
            "Entered get_data_transformer_object method of DataTransformation class"
        ) 

        try:
            logging.info("Creating ColumnTransformer from schema config")
            oh_columns = self._schema_config['oh_columns']
            or_columns = self._schema_config['or_columns']
            num_features = self._schema_config['num_features']
            log_transformer = Pipeline([
                ('log_transform', FunctionTransformer(np.log1p, validate=True))
            ])
            preprocessor = ColumnTransformer([
                ("OneHotEncoder", OneHotEncoder(handle_unknown='ignore'), oh_columns),
                ("OrdinalEncoder", OrdinalEncoder(), or_columns),
                ("LogTransformer", log_transformer, num_features),
                ("StandardScaler", StandardScaler(), num_features)
            ])


            logging.info("Created preprocessor object from ColumnTransformer")

            logging.info(
                "Exited get_data_transformer_object method of DataTransformation class"
            )
            return preprocessor
        except Exception as e:
            raise creditsException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            if self.data_validation_artifact.validation_status:
                logging.info("Starting data transformation")
                preprocessor = self.get_data_transformer_object()
                logging.info("Got the preprocessor object")
                
                train_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
                test_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.test_file_path)

                input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN])
                target_feature_train_df = train_df[TARGET_COLUMN]

                input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_test_df = test_df[TARGET_COLUMN]

                logging.info("Got train features and test features of Training dataset")


                input_feature_train_df["cb_person_default_on_file"] = input_feature_train_df["cb_person_default_on_file"].replace(
                CBPersonDefaultOnFile()._asdict())

                logging.info("mapping missing values cb_person_default_on_file of the Training dataset")

                input_feature_train_df = impute_missing_values_from_schema(input_feature_train_df, self._schema_config)
                logging.info("impute missing values of the Training dataset")

                 # Drop outliers for test set - IMPORTANT: capture both return values
                input_feature_train_df, target_feature_train_df = drop_outliers_from_schema(
                input_feature_train_df, target_feature_train_df, self._schema_config)
                logging.info("drop_outliers_from_schema values to the train dataset")
                


                input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)

                target_feature_test_df = test_df[TARGET_COLUMN]
                
                input_feature_test_df["cb_person_default_on_file"] = input_feature_test_df["cb_person_default_on_file"].replace(
                CBPersonDefaultOnFile()._asdict())

                
                logging.info("mapping missing values cb_person_default_on_file of the  Test dataset")
                
                input_feature_test_df = impute_missing_values_from_schema(input_feature_test_df, self._schema_config)
                
                logging.info("impute missing to the Test dataset")

                input_feature_test_df, target_feature_test_df = drop_outliers_from_schema(
                input_feature_test_df, target_feature_test_df, self._schema_config)
                logging.info("drop_outliers_from_schema  values to the Test dataset")
                
                target_feature_test_df = target_feature_test_df.replace(
                 CBPersonDefaultOnFile()._asdict()
                 )
                logging.info("Got train features and test features of Testing dataset")

                logging.info(
                    "Applying preprocessing object on training dataframe and testing dataframe"
                )

                input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)

                logging.info(
                    "Used the preprocessor object to fit transform the train features"
                )

                input_feature_test_arr = preprocessor.transform(input_feature_test_df)

                logging.info("Used the preprocessor object to transform the test features")

                logging.info("Applying SMOTEENN on Training dataset")

                smt = SMOTEENN(sampling_strategy="minority")

                input_feature_train_final, target_feature_train_final = smt.fit_resample(
                    input_feature_train_arr, target_feature_train_df
                )

                logging.info("Applied SMOTEENN on training dataset")

                logging.info("Applying SMOTEENN on testing dataset")

                input_feature_test_final, target_feature_test_final = smt.fit_resample(
                    input_feature_test_arr, target_feature_test_df
                )

                logging.info("Applied SMOTEENN on testing dataset")

                logging.info("Created train array and test array")

                train_arr = np.c_[
                    input_feature_train_final, np.array(target_feature_train_final.values.ravel())
                ]

                test_arr = np.c_[
                    input_feature_test_final, np.array(target_feature_test_final.values.ravel())
                ]

                save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
                save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
                save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

                logging.info("Saved the preprocessor object")

                logging.info(
                    "Exited initiate_data_transformation method of Data_Transformation class"
                )

                data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
                )
                return data_transformation_artifact
            else:
                raise Exception(self.data_validation_artifact.message)

        except Exception as e:
            raise creditsException(e, sys) from e
            

           

            
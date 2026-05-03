import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer 
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import save_object

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_object_file_path = os.path.join('models', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, numerical_columns, categorical_columns):
        '''
        This function is responsible for the data transformation
        '''
        try:
            logging.info(f"Creating pipeline for Numerical columns: {numerical_columns}")
            logging.info(f"Creating pipeline for Categorical columns: {categorical_columns}")

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', KNNImputer(n_neighbors=7)),
                    ('scaler', StandardScaler()) 
                ]
            )            

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')), 
                    ('onehot', OneHotEncoder()) 
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_pipeline, numerical_columns),
                    ('cat', cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            logging.error(CustomException(e, sys))
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed.")

            # --- Standardise column names before processing ---
            logging.info('Standardising column names (lowercase and underscores).')
            for df in [train_df, test_df]:
                df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('/', '_')

            target_column_name = 'math_score'

            # --- Automatically get column list ---
            numerical_columns = [col for col in train_df.columns if train_df[col].dtype != 'O' and col != target_column_name]
            categorical_columns = [col for col in train_df.columns if train_df[col].dtype == 'O']
            
            logging.info("Obtaining preprocessing object.")
            preprocessing_object = self.get_data_transformer_object(numerical_columns, categorical_columns)

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")
            input_feature_train_arr = preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_object.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]       

            logging.info("Saved preprocessing object.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_object_file_path,
                obj=preprocessing_object
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_object_file_path
        
        except Exception as e:
            logging.error(CustomException(e, sys))
            raise CustomException(e, sys)
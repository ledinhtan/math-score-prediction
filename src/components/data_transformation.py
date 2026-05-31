import sys

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config.configuration import DataTransformationConfig
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.data_transformation_config = config

    def get_data_transformer_object(self, numerical_columns, categorical_columns):
        try:
            logging.info("Building preprocessing pipeline for numerical columns: %s", numerical_columns)
            logging.info("Building preprocessing pipeline for categorical columns: %s", categorical_columns)

            num_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        KNNImputer(n_neighbors=self.data_transformation_config.numerical_imputer_n_neighbors),
                    ),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy=self.data_transformation_config.categorical_imputer_strategy),
                    ),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_columns),
                    ("cat", cat_pipeline, categorical_columns),
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

            logging.info("Loaded train and test datasets.")

            logging.info("Normalising datasets column names.")
            for df in [train_df, test_df]:
                df.columns = (
                    df.columns
                    .str.strip()
                    .str.lower()
                    .str.replace(" ", "_", regex=False)
                    .str.replace("/", "_", regex=False)
                )

            target_column_name = self.data_transformation_config.target_column

            numerical_columns = [
                col
                for col in train_df.columns
                if train_df[col].dtype != "O" and col != target_column_name
            ]
            categorical_columns = [col for col in train_df.columns if train_df[col].dtype == "O"]

            logging.info("Creating preprocessing object.")
            preprocessing_object = self.get_data_transformer_object(
                numerical_columns=numerical_columns,
                categorical_columns=categorical_columns,
            )

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Transforming training and testing features.")
            input_feature_train_arr = preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_object.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving preprocessing object to %s", self.data_transformation_config.preprocessor_object_file_path)
            save_object(
                file_path=self.data_transformation_config.preprocessor_object_file_path,
                obj=preprocessing_object,
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_object_file_path

        except Exception as e:
            logging.error(CustomException(e, sys))
            raise CustomException(e, sys)
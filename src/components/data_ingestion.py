import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.configuration import DataIngestionConfig
from src.exception import CustomException
from src.logger import logging


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.ingestion_config = config

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion component.")
        try:
            df = pd.read_csv(self.ingestion_config.raw_data_path)
            logging.info("Loaded raw dataset from %s", self.ingestion_config.raw_data_path)

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_output_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_output_path, index=False, header=True)

            logging.info("Running train/test split with test_size=%s random_state=%s", self.ingestion_config.test_size, self.ingestion_config.random_state)
            train_set, test_set = train_test_split(
                df,
                test_size=self.ingestion_config.test_size,
                random_state=self.ingestion_config.random_state,
            )

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed. train_path=%s, test_path=%s", self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path
        except Exception as e:
            logging.error(CustomException(e, sys))
            raise CustomException(e, sys)
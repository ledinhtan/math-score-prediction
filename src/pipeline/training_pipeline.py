import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

class TrainingPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def start_training_pipeline(self):
        try:
            logging.info('Starting training pipeline.')

            train_data_path, test_data_path = self.data_ingestion.initiate_data_ingestion()

            train_arr, test_arr, _ = self.data_transformation.initiate_data_transformation(
                train_path=train_data_path,
                test_path=test_data_path
            )

            training_result = self.model_trainer.initiate_model_trainer(
                train_array=train_arr,
                test_array=test_arr
            )

            logging.info('Training pipeline completed successfully.')

            return training_result

        except Exception as e:
            logging.error(CustomException(e, sys))
            raise CustomException(e, sys)

if __name__ == '__main__':
    pipeline = TrainingPipeline()
    result = pipeline.start_training_pipeline()
    print(result)

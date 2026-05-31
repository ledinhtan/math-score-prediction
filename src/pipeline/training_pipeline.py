import sys

import mlflow

from src.config.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging


class TrainingPipeline:
    def __init__(self):
        self.config = ConfigurationManager()
        self.data_ingestion = DataIngestion(self.config.data_ingestion_config)
        self.data_transformation = DataTransformation(self.config.data_transformation_config)
        self.model_trainer = ModelTrainer(self.config.model_trainer_config)

    def start_training_pipeline(self):
        try:
            logging.info("Starting training pipeline.")
            mlflow.set_tracking_uri(self.config.model_trainer_config.tracking_uri)
            mlflow.set_experiment(self.config.model_trainer_config.experiment_name)

            with mlflow.start_run(run_name="training_pipeline"):
                train_data_path, test_data_path = self.data_ingestion.initiate_data_ingestion()

                train_arr, test_arr, preprocessor_path = self.data_transformation.initiate_data_transformation(
                    train_path=train_data_path,
                    test_path=test_data_path,
                )

                mlflow.log_artifact(preprocessor_path)

                training_result = self.model_trainer.initiate_model_trainer(
                    train_array=train_arr,
                    test_array=test_arr,
                )

                logging.info("Training pipeline completed successfully.")
                return training_result

        except Exception as e:
            logging.error(CustomException(e, sys))
            raise CustomException(e, sys)


if __name__ == '__main__':
    pipeline = TrainingPipeline()
    print(pipeline.start_training_pipeline())
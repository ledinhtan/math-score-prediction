import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.utils import read_yaml

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
CONFIG_FILE_PATH = os.path.join(ROOT_DIR, "config", "config.yaml")

@dataclass
class DataIngestionConfig:
    raw_data_path: str
    raw_data_output_path: str
    train_data_path: str
    test_data_path: str
    test_size: float
    random_state: int

@dataclass
class DataTransformationConfig:
    preprocessor_object_file_path: str
    target_column: str
    numerical_imputer_n_neighbors: int
    categorical_imputer_strategy: str

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str
    test_size: float
    random_state: int
    validation_split: float
    optuna_n_trials: int
    optuna_timeout: int
    n_jobs: int
    optuna_sampler: str
    optuna_pruner_enabled: bool
    optuna_pruner_n_startup_trials: int
    optuna_pruner_n_warmup_steps: int
    experiment_name: str
    tracking_uri: str

class ConfigurationManager:
    def __init__(self, config_filepath: str = None):
        if config_filepath is None:
            config_filepath = CONFIG_FILE_PATH

        self.config_info = read_yaml(config_filepath)
        self.data_ingestion_config = self._get_data_ingestion_config()
        self.data_transformation_config = self._get_data_transformation_config()
        self.model_trainer_config = self._get_model_trainer_config()

    def _get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            config = self.config_info.get("artifacts", {})
            training_conf = self.config_info.get("training", {})
            return DataIngestionConfig(
                raw_data_path=config["raw_data_path"],
                raw_data_output_path=config["raw_data_output_path"],
                train_data_path=config["train_data_path"],
                test_data_path=config["test_data_path"],
                test_size=training_conf["test_size"],
                random_state=training_conf["random_state"],
            )
        except Exception as e:
            raise CustomException(e, sys)

    def _get_data_transformation_config(self) -> DataTransformationConfig:
        try:
            model_conf = self.config_info.get("model", {})
            preprocessing_conf = self.config_info.get("preprocessing", {})
            return DataTransformationConfig(
                preprocessor_object_file_path=model_conf["preprocessor_file"],
                target_column=model_conf["target_column"],
                numerical_imputer_n_neighbors=preprocessing_conf["numerical_imputer_n_neighbors"],
                categorical_imputer_strategy=preprocessing_conf["categorical_imputer_strategy"],
            )
        except Exception as e:
            raise CustomException(e, sys)

    def _get_model_trainer_config(self) -> ModelTrainerConfig:
        try:
            model_conf = self.config_info.get("model", {})
            training_conf = self.config_info.get("training", {})
            experiment_conf = self.config_info.get("experiment", {})

            return ModelTrainerConfig(
                trained_model_file_path=model_conf["trained_model_file"],
                test_size=training_conf["test_size"],
                random_state=training_conf["random_state"],
                validation_split=training_conf["validation_split"],
                optuna_n_trials=training_conf["optuna"]["n_trials"],
                optuna_timeout=training_conf["optuna"]["timeout"],
                n_jobs=training_conf["optuna"]["n_jobs"],
                optuna_sampler=training_conf["optuna"]["sampler"],
                optuna_pruner_enabled=training_conf["optuna"]["pruner"]["enabled"],
                optuna_pruner_n_startup_trials=training_conf["optuna"]["pruner"]["n_startup_trials"],
                optuna_pruner_n_warmup_steps=training_conf["optuna"]["pruner"]["n_warmup_steps"],
                experiment_name=experiment_conf["name"],
                tracking_uri=experiment_conf["tracking_uri"],
            )
        except Exception as e:
            raise CustomException(e, sys)
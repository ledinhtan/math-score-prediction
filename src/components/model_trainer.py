import os
import sys
from dataclasses import dataclass

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score


from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('models', "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data.")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Support Vector Machine": SVR(),
                "k-Nearest Neighbours": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Random Forest": RandomForestRegressor(),
                "XGBRegressor": xgb.XGBRegressor(),
            }

            params = {
                "Linear Regression": {},
                "Support Vector Machine": {
                    # 'kernel': ['rbf', 'sigmoid', 'poly', 'linear'], 
                    'C': np.arange(3.0, 6.1, 0.1)
                },
                "k-Nearest Neighbours": {
                    'weights': ['uniform', 'distance'],
                    # 'n_neighbors': list(range(1, 20))
                },
                "Decision Tree": {
                    # 'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter': ['best', 'random'],
                    'max_features': ['sqrt', 'log2'],
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    # 'learning_rate': [0.01, 0.05, 0.1],
                    # 'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    # 'learning_rate': [.1, .01, 0.5, .001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    # 'learning_rate': [.1, .01, .05, .001],
                    # 'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    # 'criterion': ['squared_error', 'friedman_mse'],
                    # 'max_features': ['auto', 'sqrt', 'log2'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Random Forest": {
                    # 'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features': ['sqrt', 'log2', None],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "XGBRegressor": {
                    # 'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                logging.error(f"Critical: Model score {best_model_score} is unacceptable.")
                raise CustomException("No best model found.")
            elif best_model_score < 0.8:
                logging.warning(f"Warning: Model score {best_model_score} is a bit low. Performance might be unstable.")

            logging.info("Best found model on both training and testing dataset.")
            logging.info(f"Best model is {best_model_name}.")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            y_predicted = best_model.predict(X_test)

            r2_scores = r2_score(y_test, y_predicted)

            return r2_scores

        except Exception as e:
            logging.error(CustomException(e, sys))
            raise CustomException(e, sys)
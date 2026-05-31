import sys

import mlflow
import optuna
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.config.configuration import ModelTrainerConfig
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        mlflow.set_tracking_uri(self.config.tracking_uri)
        mlflow.set_experiment(self.config.experiment_name)

    def _build_model(self, trial):
        model_name = trial.suggest_categorical(
            "model_name",
            [
                "linear_regression",
                "svr",
                "knn",
                "decision_tree",
                "catboost",
                "adaboost",
                "gradient_boosting",
                "random_forest",
                "xgboost",
            ],
        )

        if model_name == "linear_regression":
            model = LinearRegression(
                fit_intercept=trial.suggest_categorical("lr_fit_intercept", [True, False]),
            )
        elif model_name == "svr":
            model = SVR(
                C=trial.suggest_float("svr_C", 0.1, 10.0, log=True),
                epsilon=trial.suggest_float("svr_epsilon", 0.01, 1.0),
                gamma=trial.suggest_categorical("svr_gamma", ["scale", "auto"]),
                kernel=trial.suggest_categorical("svr_kernel", ["rbf", "poly", "sigmoid", "linear"]),
            )
        elif model_name == "knn":
            model = KNeighborsRegressor(
                n_neighbors=trial.suggest_int("knn_n_neighbors", 1, 20),
                weights=trial.suggest_categorical("knn_weights", ["uniform", "distance"]),
                p=trial.suggest_categorical("knn_p", [1, 2]),
            )
        elif model_name == "decision_tree":
            model = DecisionTreeRegressor(
                criterion=trial.suggest_categorical(
                    "dt_criterion",
                    ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                ),
                splitter=trial.suggest_categorical("dt_splitter", ["best", "random"]),
                max_depth=trial.suggest_int("dt_max_depth", 2, 20),
                min_samples_split=trial.suggest_int("dt_min_samples_split", 2, 10),
                max_features=trial.suggest_categorical("dt_max_features", [None, "sqrt", "log2"]),
                random_state=self.config.random_state,
            )
        elif model_name == "catboost":
            model = CatBoostRegressor(
                iterations=trial.suggest_int("catboost_iterations", 50, 500, step=50),
                depth=trial.suggest_int("catboost_depth", 4, 10),
                learning_rate=trial.suggest_float("catboost_learning_rate", 0.01, 0.2, log=True),
                l2_leaf_reg=trial.suggest_float("catboost_l2_leaf_reg", 1.0, 10.0),
                verbose=False,
                random_seed=self.config.random_state,
            )
        elif model_name == "adaboost":
            model = AdaBoostRegressor(
                n_estimators=trial.suggest_int("ada_n_estimators", 10, 300, step=10),
                learning_rate=trial.suggest_float("ada_learning_rate", 0.01, 1.0, log=True),
                loss=trial.suggest_categorical("ada_loss", ["linear", "square", "exponential"]),
                random_state=self.config.random_state,
            )
        elif model_name == "gradient_boosting":
            model = GradientBoostingRegressor(
                loss=trial.suggest_categorical(
                    "gb_loss", ["squared_error", "huber", "absolute_error", "quantile"]
                ),
                learning_rate=trial.suggest_float("gb_learning_rate", 0.01, 0.3, log=True),
                n_estimators=trial.suggest_int("gb_n_estimators", 50, 300, step=50),
                subsample=trial.suggest_float("gb_subsample", 0.5, 1.0),
                criterion=trial.suggest_categorical("gb_criterion", ["friedman_mse", "squared_error"]),
                max_depth=trial.suggest_int("gb_max_depth", 3, 12),
                max_features=trial.suggest_categorical("gb_max_features", [None, "sqrt", "log2"]),
                random_state=self.config.random_state,
            )
        elif model_name == "random_forest":
            model = RandomForestRegressor(
                n_estimators=trial.suggest_int("rf_n_estimators", 50, 300, step=50),
                criterion=trial.suggest_categorical(
                    "rf_criterion", ["squared_error", "friedman_mse", "absolute_error", "poisson"]
                ),
                max_depth=trial.suggest_int("rf_max_depth", 4, 20),
                min_samples_split=trial.suggest_int("rf_min_samples_split", 2, 10),
                max_features=trial.suggest_categorical("rf_max_features", [None, "sqrt", "log2"]),
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
            )
        else:
            model = XGBRegressor(
                n_estimators=trial.suggest_int("xgb_n_estimators", 50, 300, step=50),
                max_depth=trial.suggest_int("xgb_max_depth", 3, 12),
                learning_rate=trial.suggest_float("xgb_learning_rate", 0.01, 0.2, log=True),
                subsample=trial.suggest_float("xgb_subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("xgb_colsample_bytree", 0.5, 1.0),
                gamma=trial.suggest_float("xgb_gamma", 0.0, 5.0),
                reg_lambda=trial.suggest_float("xgb_reg_lambda", 1e-3, 10.0, log=True),
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                verbosity=0,
            )

        return model_name, model

    def _build_model_from_params(self, params):
        model_name = params["model_name"]

        if model_name == "linear_regression":
            return LinearRegression(
                fit_intercept=params["lr_fit_intercept"],
            )

        if model_name == "svr":
            return SVR(
                C=params["svr_C"],
                epsilon=params["svr_epsilon"],
                gamma=params["svr_gamma"],
                kernel=params["svr_kernel"],
            )

        if model_name == "knn":
            return KNeighborsRegressor(
                n_neighbors=params["knn_n_neighbors"],
                weights=params["knn_weights"],
                p=params["knn_p"],
            )

        if model_name == "decision_tree":
            return DecisionTreeRegressor(
                criterion=params["dt_criterion"],
                splitter=params["dt_splitter"],
                max_depth=params["dt_max_depth"],
                min_samples_split=params["dt_min_samples_split"],
                max_features=params["dt_max_features"],
                random_state=self.config.random_state,
            )

        if model_name == "catboost":
            return CatBoostRegressor(
                iterations=params["catboost_iterations"],
                depth=params["catboost_depth"],
                learning_rate=params["catboost_learning_rate"],
                l2_leaf_reg=params["catboost_l2_leaf_reg"],
                verbose=False,
                random_seed=self.config.random_state,
            )

        if model_name == "adaboost":
            return AdaBoostRegressor(
                n_estimators=params["ada_n_estimators"],
                learning_rate=params["ada_learning_rate"],
                loss=params["ada_loss"],
                random_state=self.config.random_state,
            )

        if model_name == "gradient_boosting":
            return GradientBoostingRegressor(
                loss=params["gb_loss"],
                learning_rate=params["gb_learning_rate"],
                n_estimators=params["gb_n_estimators"],
                subsample=params["gb_subsample"],
                criterion=params["gb_criterion"],
                max_depth=params["gb_max_depth"],
                max_features=params["gb_max_features"],
                random_state=self.config.random_state,
            )

        if model_name == "random_forest":
            return RandomForestRegressor(
                n_estimators=params["rf_n_estimators"],
                criterion=params["rf_criterion"],
                max_depth=params["rf_max_depth"],
                min_samples_split=params["rf_min_samples_split"],
                max_features=params["rf_max_features"],
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
            )

        return XGBRegressor(
            n_estimators=params["xgb_n_estimators"],
            max_depth=params["xgb_max_depth"],
            learning_rate=params["xgb_learning_rate"],
            subsample=params["xgb_subsample"],
            colsample_bytree=params["xgb_colsample_bytree"],
            gamma=params["xgb_gamma"],
            reg_lambda=params["xgb_reg_lambda"],
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            verbosity=0,
        )

    def _objective(self, trial, X_train, y_train, X_valid, y_valid):
        model_name, model = self._build_model(trial)

        with mlflow.start_run(nested=True):
            mlflow.log_param("model_name", model_name)
            mlflow.log_params(trial.params)

            model.fit(X_train, y_train)
            predictions = model.predict(X_valid)
            mse = mean_squared_error(y_valid, predictions)
            r2 = r2_score(y_valid, predictions)

            mlflow.log_metric("mse_validation", mse)
            mlflow.log_metric("r2_validation", r2)

        return mse

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Preparing training arrays for Optuna search.")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            X_train_opt, X_valid_opt, y_train_opt, y_valid_opt = train_test_split(
                X_train,
                y_train,
                test_size=self.config.validation_split,
                random_state=self.config.random_state,
            )

            sampler = (
                optuna.samplers.TPESampler()
                if self.config.optuna_sampler.lower() == "tpe"
                else optuna.samplers.RandomSampler()
            )
            pruner = (
                optuna.pruners.MedianPruner(
                    n_startup_trials=self.config.optuna_pruner_n_startup_trials,
                    n_warmup_steps=self.config.optuna_pruner_n_warmup_steps,
                )
                if self.config.optuna_pruner_enabled
                else optuna.pruners.NopPruner()
            )

            study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
            study.optimize(
                lambda trial: self._objective(trial, X_train_opt, y_train_opt, X_valid_opt, y_valid_opt),
                n_trials=self.config.optuna_n_trials,
                timeout=self.config.optuna_timeout,
            )

            best_params = study.best_trial.params
            best_model_name = best_params["model_name"]
            best_model = self._build_model_from_params(best_params)

            logging.info("Retraining best model (%s) on full training data.", best_model_name)
            best_model.fit(X_train, y_train)

            y_pred = best_model.predict(X_test)
            test_mse = mean_squared_error(y_test, y_pred)
            test_r2 = r2_score(y_test, y_pred)

            save_object(file_path=self.config.trained_model_file_path, obj=best_model)

            mlflow.log_param("best_model", best_model_name)
            mlflow.log_metric("mse_test", float(test_mse))
            mlflow.log_metric("r2_test", float(test_r2))
            mlflow.log_artifact(self.config.trained_model_file_path)

            return {
                "best_model": best_model_name,
                "mse_test": float(test_mse),
                "r2_test": float(test_r2),
                "model_path": self.config.trained_model_file_path,
            }

        except Exception as e:
            logging.error(CustomException(e, sys))
            raise CustomException(e, sys)
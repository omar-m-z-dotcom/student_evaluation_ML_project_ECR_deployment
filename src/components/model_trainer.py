import os
from src.exception import CustomException
from src.logger import logging
import numpy as np
from src.utils import save_model, evaluate_models_with_tuning
from dataclasses import dataclass
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        """
        Constructor for the ModelTrainer class.
        Initializes the ModelTrainerConfig object as an instance variable
        and sets the model to None.
        """
        self.model_trainer_config = ModelTrainerConfig()
        self.model = None

    def initiate_model_trainer(self, train_array: np.ndarray, test_array: np.ndarray):
        """
        This function initiates the model training process. It first splits the given train and test data into input and target features.
        Then it creates a dictionary of models to be trained and a dictionary of hyperparameter distributions for each model.
        The function then calls the evaluate_models_with_tuning function to train each model with its hyperparameters and evaluate its performance.
        It then finds the model with the highest test R2 score and checks if it is overfitting the data.
        If the model is not overfitting, the function saves the model and returns its test R2 score.
        If the model is overfitting, the function raises a CustomException.
        """
        try:
            logging.info("Splitting training and test input and target feature")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            models = {
                "Random Forest": RandomForestRegressor,
                "Decision Tree": DecisionTreeRegressor,
                "Gradient Boosting": GradientBoostingRegressor,
                "Linear Regression": LinearRegression,
                "K-Neighbors Regressor": KNeighborsRegressor,
                "XGBRegressor": XGBRegressor,
                "CatBoosting Regressor": CatBoostRegressor,
                "AdaBoost Regressor": AdaBoostRegressor,
            }
            param_spaces = {
                "Random Forest": {
                    "n_estimators": {"type": "int", "low": 50, "high": 300},
                    "max_depth": {"type": "int", "low": 3, "high": 15},
                    "min_samples_split": {"type": "int", "low": 2, "high": 20},
                },
                "Decision Tree": {"max_depth": {"type": "int", "low": 3, "high": 15}},
                "Gradient Boosting": {
                    "n_estimators": {"type": "int", "low": 50, "high": 300},
                    "learning_rate": {
                        "type": "float",
                        "low": 0.01,
                        "high": 0.3,
                        "log": True,
                    },
                    "max_depth": {"type": "int", "low": 3, "high": 10},
                },
                "Linear Regression": {},
                "K-Neighbors Regressor": {
                    "n_neighbors": {"type": "int", "low": 2, "high": 20}
                },
                "XGBRegressor": {
                    "n_estimators": {"type": "int", "low": 50, "high": 300},
                    "learning_rate": {
                        "type": "float",
                        "low": 0.01,
                        "high": 0.3,
                        "log": True,
                    },
                    "max_depth": {"type": "int", "low": 3, "high": 10},
                },
                "CatBoosting Regressor": {
                    "iterations": {"type": "int", "low": 50, "high": 300},
                    "learning_rate": {
                        "type": "float",
                        "low": 0.01,
                        "high": 0.3,
                        "log": True,
                    },
                    "depth": {"type": "int", "low": 3, "high": 10},
                },
                "AdaBoost Regressor": {
                    "n_estimators": {"type": "int", "low": 50, "high": 300},
                    "learning_rate": {
                        "type": "float",
                        "low": 0.01,
                        "high": 0.3,
                        "log": True,
                    },
                },
            }

            evaluation_results = evaluate_models_with_tuning(
                X_train, y_train, X_test, y_test, models, param_spaces
            )
            logging.info("Model training completed")
            best_model_test_score = max(
                [evaluation_results[model][1] for model in evaluation_results.keys()]
            )
            if best_model_test_score <= 0.6:
                logging.info("No good model found with test R2 score more than 0.6")
                raise CustomException(
                    "No good model found with test R2 score more than 0.6"
                )
            # find the train dataset results for the best model
            for model in evaluation_results.keys():
                if evaluation_results[model][1] == best_model_test_score:
                    best_model_train_score = evaluation_results[model][0]
                    if abs(best_model_test_score - best_model_train_score) > 0.05:
                        logging.info("the best model is overfitting the data")
                        raise CustomException("the best model is overfitting the data")
                    self.model = evaluation_results[model][2]
                    logging.info(
                        f"Best model found: {model} with test R2 score: {best_model_test_score} and train R2 score: {best_model_train_score}"
                    )
                    break
            predictions = self.model.predict(X_test)
            r2_square = r2_score(y_test, predictions)
            logging.info(f"Best model R2 Score on test data: {r2_square}")
            logging.info("Saving the model")
            save_model(self.model, self.model_trainer_config.trained_model_file_path)
            logging.info("Model saved")
            return best_model_test_score
        except Exception as e:
            logging.exception(e)
            raise CustomException(e)

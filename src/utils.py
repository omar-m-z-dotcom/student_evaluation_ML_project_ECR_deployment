# import os
# import pandas as pd
# import numpy as np
import pickle
from src.exception import CustomException
from src.logger import logging
import numpy as np
from sklearn.metrics import r2_score
import optuna
from optuna.trial import Trial


def save_model(model, model_path: str):
    """
    This function saves a given model to a specified path.

    Args:
        model (object): The model to be saved.
        model_path (str): The path where the model is to be saved.

    Raises:
        CustomException: If there is an error during the execution of the function.
    """
    try:
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    except Exception as e:
        logging.exception(e)
        raise CustomException(e)


def load_model(model_path: str):
    """
    This function loads a model from a specified path.

    Args:
        model_path (str): The path where the model is saved.

    Returns:
        object: The loaded model.

    Raises:
        CustomException: If there is an error during the execution of the function.
    """
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logging.exception(e)
        raise CustomException(e)


def optimize_hyperparameters(
    model_class,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    param_distributions: dict,
    n_trials=100,
):
    """
    This function performs hyperparameter optimization for a given model class using Optuna.
    It takes as input the model class, the training and test data, a dictionary of hyperparameter distributions,
    and the number of trials to run.

    The function first defines an objective function, which constructs a dictionary of hyperparameters from a Trial object,
    creates an instance of the given model class with the given hyperparameters, fits the model to the training data,
    makes predictions on the test data, and computes the R2 score of the predictions.

    The function then runs the hyperparameter optimization using Optuna, and returns the best hyperparameters, the best score,
    and the best model.

    Args:
        model_class (class): The class of the model to optimize.
        x_train (array): The training data input feature.
        y_train (array): The training data target feature.
        x_test (array): The test data input feature.
        y_test (array): The test data target feature.
        param_distributions (dict): A dictionary of hyperparameter distributions.
        n_trials (int): The number of trials to run.

    Returns:
        tuple: A tuple containing the best hyperparameters, the best score, and the best model.
    """

    def objective(trial: Trial) -> float:
        """
        This function is the objective function for the hyperparameter optimization
        using Optuna. It takes a Trial object as input, and returns a float value
        representing the score of the model with the given hyperparameters.

        The function first constructs a dictionary of hyperparameters from the
        Trial object. It then creates an instance of the given model class
        with the given hyperparameters, fits the model to the training data, makes
        predictions on the test data, and computes the R2 score of the predictions.

        If there is an error during the execution of the function, it logs the
        exception and raises a CustomException.

        Args:
            trial (Trial): The Trial object containing the hyperparameters to be tried.

        Returns:
            float: The R2 score of the model with the given hyperparameters.
        """
        try:
            params = {}
            for param_name, param_config in param_distributions.items():
                param_type = param_config["type"]

                if param_type == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config["values"]
                    )
                elif param_type == "int":
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config["low"],
                        param_config["high"],
                        step=param_config.get("step", 1),
                    )
                elif param_type == "float":
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config["low"],
                        param_config["high"],
                        log=param_config.get("log", False),
                    )
            logging.info(f"for model {model_class.__name__} Trying params: {params}")
            model = model_class(**params)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            score = r2_score(y_test, y_pred)
            return score
        except Exception as e:
            logging.exception(e)
            raise CustomException(e)

    try:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        best_model = model_class(**best_params)
        best_model.fit(x_train, y_train)
        best_score = study.best_value
        return best_params, best_score, best_model
    except Exception as e:
        logging.exception(e)
        raise CustomException(e)


def evaluate_models_with_tuning(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    models: dict,
    param_spaces: dict,
    n_trials=50,
):
    """
    Evaluate the performance of multiple models with hyperparameter tuning.

    Args:
        x_train (np.ndarray): The input features of the training data.
        y_train (np.ndarray): The target feature of the training data.
        x_test (np.ndarray): The input features of the testing data.
        y_test (np.ndarray): The target feature of the testing data.
        models (dict): A dictionary of model classes to be evaluated.
        param_spaces (dict): A dictionary of hyperparameter distributions for each model.
        n_trials (int): The number of hyperparameter combinations to try.

    Returns:
        dict: A dictionary containing the performance metrics of each model. The keys are the model names, and the values are tuples containing the training R2 score, the test R2 score, and the model instance.

    Raises:
        CustomException: If there is an error during the execution of the function.
    """

    try:
        report = {}
        for model_name, model_class in models.items():
            if model_name in param_spaces:
                logging.info(f"Optimizing hyperparameters for {model_name}...")
                best_params, best_score, best_model = optimize_hyperparameters(
                    model_class,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    param_spaces[model_name],
                    n_trials,
                )
                y_train_pred = best_model.predict(x_train)
                train_model_score = r2_score(y_train, y_train_pred)

                report[model_name] = (train_model_score, best_score, best_model)
                logging.info(
                    f"{model_name} - Best params: {best_params}, Test R2: {best_score}"
                )
            else:
                # Fall back to default evaluation if no param space is provided
                model = model_class()
                model.fit(x_train, y_train)
                y_train_pred = model.predict(x_train)
                y_test_pred = model.predict(x_test)
                train_model_score = r2_score(y_train, y_train_pred)
                test_model_score = r2_score(y_test, y_test_pred)
                report[model_name] = (train_model_score, test_model_score, model)

        return report
    except Exception as e:
        logging.exception(e)
        raise CustomException(e)

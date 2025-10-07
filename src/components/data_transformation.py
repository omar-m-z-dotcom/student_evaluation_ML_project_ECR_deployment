import os
from src.exception import CustomException
from src.logger import logging
from src.utils import save_model
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig

    def get_data_transformation_object(self):
        """
        This function creates a data transformation object which is a pipeline
        composed of two sub-pipelines for numerical and categorical features
        respectively. The numerical pipeline consists of imputation of missing
        values using median and scaling using StandardScaler. The categorical
        pipeline consists of imputation of missing values using most frequent
        value, one-hot encoding and scaling using StandardScaler with mean
        set to False. The data transformation object is then returned.

        Returns:
            ColumnTransformer: The data transformation object.
        """
        try:
            numerical_columns = ["writing score", "reading score"]
            categorical_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course",
            ]
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )
            logging.info(
                f"Numerical columns: {numerical_columns} and numerical pipline is completed"
            )
            logging.info(
                f"Categorical columns: {categorical_columns} and categorical pipline is completed"
            )
            return preprocessor
        except Exception as e:
            logging.exception(e)
            raise CustomException(e)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        """
        This function reads the training and testing data from their respective paths, drops the target column,
        applies the preprocessing object to the input features of the training and testing data, and
        returns the preprocessed training and testing data arrays along with the path to the preprocessing object file.
        Args:
            train_path (str): The path to the training data file.
            test_path (str): The path to the testing data file.
        Returns:
            tuple: A tuple containing the preprocessed training data array, the preprocessed testing data array, and the path to the preprocessing object file.
        Raises:
            CustomException: If there is an error during the execution of the function.
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformation_object()
            target_column_name = "math score"
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            train_arr: np.ndarray = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr: np.ndarray = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            logging.info(f"processing completed")
            save_model(
                preprocessing_obj,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            logging.info(f"Preprocessor pickle file saved")
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            logging.exception(e)
            raise CustomException(e)

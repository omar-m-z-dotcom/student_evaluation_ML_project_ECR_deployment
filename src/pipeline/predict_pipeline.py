import pandas as pd
from src.exception import CustomException
from src.utils import load_model
from src.logger import logging
from sklearn.compose import ColumnTransformer
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig


class PredictPipeline:
    def __init__(self):
        """
        Constructor for the PredictPipeline class.
        Initializes the PredictPipeline object with no arguments.
        """
        pass

    def predict(self, features):
        """
        Predicts the math score given the input features.

        Args:
            features (pandas.DataFrame): A DataFrame containing the input features.

        Returns:
            preds (numpy.ndarray): An array containing the predicted math scores.

        Raises:
            CustomException: If there is an error during the execution of the function.
        """
        try:
            model_path = ModelTrainerConfig.trained_model_file_path
            preprocessor_path = DataTransformationConfig.preprocessor_obj_file_path
            model = load_model(model_path)
            preprocessor: ColumnTransformer = load_model(preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            logging.exception(e)
            raise CustomException(e)


class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int,
    ):
        """
        Initializes a CustomData object with the given parameters.

        Parameters:
            gender (str): The gender of the student.
            race_ethnicity (str): The race/ethnicity of the student.
            parental_level_of_education (str): The parental level of education.
            lunch (str): The lunch type of the student.
            test_preparation_course (str): Whether the student took a test preparation course or not.
            reading_score (int): The reading score of the student.
            writing_score (int): The writing score of the student.
        """
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        """
        This function returns a pandas DataFrame containing the input data
        for prediction. It takes the input parameters of the CustomData object
        and creates a dictionary with the parameter names as keys and
        the parameter values as values in lists. The dictionary is then
        used to create a pandas DataFrame which is returned.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the input data for prediction.

        Raises:
            CustomException: If there is an error during the execution of the function.
        """
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race/ethnicity": [self.race_ethnicity],
                "parental level of education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test preparation course": [self.test_preparation_course],
                "reading score": [self.reading_score],
                "writing score": [self.writing_score],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            logging.exception(e)
            raise CustomException(e)

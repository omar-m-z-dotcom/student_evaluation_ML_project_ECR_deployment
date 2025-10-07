import os
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# from src.components.data_transformation import DataTransformation
# from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        """
        Constructor for the DataIngestion class.
        Initializes the DataIngestionConfig object as an instance variable.
        """
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        This function initiates the data ingestion process. It logs into Kaggle Hub
        to download the dataset, reads it into a pandas DataFrame, saves it to a
        raw data file, splits the data into training and test sets using the
        train_test_split function from scikit-learn and saves the training and test data
        to separate files.

        Returns:
            tuple: A tuple containing the paths to the raw data file, the training data file and the test data file.

        Raises:
            CustomException: If there is an error during the execution of the function.
        """
        logging.info("Entered the data ingestion method or component")
        try:
            kagglehub.login()
            file_path = kagglehub.dataset_download(
                "spscientist/students-performance-in-exams", "StudentsPerformance.csv"
            )
            df = pd.read_csv(file_path)
            logging.info("Read the dataset as dataframe")
            os.makedirs(
                os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True
            )
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )
            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.raw_data_path,
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            logging.exception(e)
            raise CustomException(e)


# if __name__ == "__main__":
#     obj = DataIngestion()
#     _, train_data_path, test_data_path = obj.initiate_data_ingestion()
#     data_transformation = DataTransformation()
#     train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
#         train_data_path, test_data_path
#     )
#     model_trainer = ModelTrainer()
#     r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
#     print(f"R2 score: {r2_score}")

import pandas as pd
import sys
from ChurnPrediction.logger import logger
from ChurnPrediction.exception import CustomException
import os

class DataIngestion:
    def __init__(self, file_path=None):
        if file_path:
            self.file_path = file_path
        else:
            self.file_path = os.path.join(os.getcwd(), "src", "ChurnPrediction", "data", "TelcoDataset.csv")


    def load_data(self):
        try:
            logger.info(f"Reading dataset from: {self.file_path}")
            df = pd.read_csv(self.file_path)
            logger.info("Data loaded successfully.")
            return df
        except Exception as e:
            raise CustomException(e, sys)

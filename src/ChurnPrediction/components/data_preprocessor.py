import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler
from ChurnPrediction.logger import logger
from ChurnPrediction.exception import CustomException

class DataPreprocessor:
    def preprocess(self, df):
        try:
            df = pd.get_dummies(df, columns=[
                'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                'PaperlessBilling', 'PaymentMethod'
            ], drop_first=True)

            scaler = StandardScaler()
            numerical_columns = ['SeniorCitizen', 'tenure', 'MonthlyCharges']
            df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

            X = df[numerical_columns]
            y = (df['Churn'] == 'Yes').astype(int)

            logger.info("Preprocessing completed.")
            return X, y, scaler
        except Exception as e:
            raise CustomException(e, sys)

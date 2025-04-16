import sys 
from sklearn.metrics import accuracy_score, f1_score
from ChurnPrediction.logger import logger
from ChurnPrediction.exception import CustomException

class ModelEvaluator:
    def evaluate_model(self, model, X_test, y_test):
        try:
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            logger.info("Model evaluation done.")
            return {"accuracy": acc, "f1_score": f1}
        except Exception as e:
            raise CustomException(e, sys)

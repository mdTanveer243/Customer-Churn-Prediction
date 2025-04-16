import sys
from ChurnPrediction.components.data_ingestion import DataIngestion
from ChurnPrediction.components.data_preprocessor import DataPreprocessor
from ChurnPrediction.components.model_trainer import ModelTrainer
from ChurnPrediction.components.model_evaluator import ModelEvaluator

from ChurnPrediction.logger import logger
from ChurnPrediction.exception import CustomException

from sklearn.model_selection import train_test_split

def start_training():
    try:
        # Step 1: Load data
        data_ingestor = DataIngestion()
        df = data_ingestor.load_data()

        # Step 2: Preprocess data
        preprocessor = DataPreprocessor()
        X, y, scaler = preprocessor.preprocess(df)

        # Step 3: Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Step 4: Train and Evaluate Models
        trainer = ModelTrainer()
        models, param_grids = trainer.get_models_and_params()

        results = []

        for name, model in models.items():
            trained_model, best_params = trainer.train_model(name, model, param_grids.get(name, {}), X_train, y_train)

            evaluator = ModelEvaluator()
            eval_metrics = evaluator.evaluate_model(trained_model, X_test, y_test)

            results.append({
                "Model": name,
                "Accuracy": eval_metrics["accuracy"],
                "F1 Score": eval_metrics["f1_score"]
            })

            logger.info(f"{name} - Accuracy: {eval_metrics['accuracy']} | F1 Score: {eval_metrics['f1_score']}")

        return results
    
    except Exception as e:
        raise CustomException(e, sys)

import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from ChurnPrediction.pipelines.training_pipeline import start_training



if __name__ == "__main__":
    results = start_training()
    for r in sorted(results, key=lambda x: x["Accuracy"], reverse=True):
        print(r)

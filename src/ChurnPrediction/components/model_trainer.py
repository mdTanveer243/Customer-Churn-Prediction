import sys
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from ChurnPrediction.logger import logger
from ChurnPrediction.exception import CustomException

class ModelTrainer:
    def get_models_and_params(self):
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42),
            'Support Vector Machine': SVC(),
            'Naive Bayes': GaussianNB(),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Decision Tree': DecisionTreeClassifier()
        }

        param_grids = {
            'Logistic Regression': {
                'C': [0.01, 0.1, 1, 10],
                'solver': ['liblinear', 'lbfgs']
            },
            'Random Forest': {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt']
            },
            'Support Vector Machine': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            },
            'Naive Bayes': {},
            'K-Nearest Neighbors': {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            },
            'Decision Tree': {
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'criterion': ['gini', 'entropy']
            }
        }

        return models, param_grids

    def train_model(self, name, model, param_grid, X_train, y_train):
        try:
            if param_grid:
                grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
            else:
                model.fit(X_train, y_train)
                best_model = model
                best_params = {}

            logger.info(f"{name} training completed with best params: {best_params}")
            return best_model, best_params
        except Exception as e:
            raise CustomException(e, sys)

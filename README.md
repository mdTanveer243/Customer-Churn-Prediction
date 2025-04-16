# Customer Churn Prediction

This project is focused on predicting customer churn in a telecommunications company using machine learning algorithms. The goal is to predict whether a customer will leave (churn) or stay based on features such as customer demographics, account information, and usage data.

## Project Overview

The **Customer Churn Prediction** pipeline includes the following main steps:

1. **Data Ingestion**: Load the customer data from a CSV file.
2. **Data Preprocessing**: Clean and transform the data into a suitable format for training.
3. **Model Training**: Train multiple machine learning models to predict churn.
4. **Model Evaluation**: Evaluate the models based on performance metrics like Accuracy and F1-Score.
5. **Hyperparameter Tuning**: Optimize model parameters for better performance.

The goal is to provide actionable insights for businesses to identify customers at risk of leaving.

## Key Features

- **Data Preprocessing**: Handling missing values, encoding categorical variables, and scaling numerical features.
- **Model Training**: Implemented several classification algorithms such as Logistic Regression, Support Vector Machine, Random Forest, and more.
- **Model Evaluation**: Evaluate models using metrics like Accuracy and F1-Score to compare their performance.
- **Logging and Exception Handling**: Integrated logging for monitoring and custom exception handling for debugging.

## Results

The models were trained and evaluated on the dataset, and the following results were obtained:

| Model                     | Accuracy  | F1 Score  |
|---------------------------|-----------|-----------|
| Logistic Regression        | 80.77%    | 57.19%    |
| Support Vector Machine     | 80.34%    | 56.24%    |
| Random Forest              | 79.35%    | 52.99%    |
| K-Nearest Neighbors        | 78.21%    | 54.52%    |
| Decision Tree              | 77.50%    | 49.44%    |
| Naive Bayes                | 76.15%    | 38.01%    |

The **Logistic Regression** model achieved the highest accuracy (80.77%) and a reasonable F1 Score (57.19%).

## Getting Started

### Prerequisites

To run the project, you will need the following:

- Python 3.x
- Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `logging`, `joblib`, and other dependencies listed in `requirements.txt`.

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/mdTanveer243/Customer-Churn-Prediction.git
   cd Customer-Churn-Prediction

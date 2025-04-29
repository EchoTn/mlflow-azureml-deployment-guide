"""
Train and register an Iris flower classification model using MLflow and scikit-learn.

This script performs the following steps:
1. Loads the Iris dataset as a pandas DataFrame.
2. Splits the dataset into training and test sets.
3. Applies feature standardization using `StandardScaler`.
4. Trains a logistic regression model on the training data.
5. Uses MLflow to log parameters, track the run, and register the trained model in the Azure ML model registry.

Functions:
- `load_and_preprocess_data()`:
    Loads the Iris dataset from scikit-learn, applies preprocessing (scaling),
    and returns DataFrame-formatted train and test sets with feature column names preserved.

- `train_and_log_model(X_train, y_train)`:
    Trains a logistic regression model on the provided training data, logs the hyperparameters using MLflow,
    and registers the trained model as "iris-model" in the registry.

- `main()`:
    Orchestrates the training flow by calling data loading/preprocessing and model training functions.

Configuration:
- The script sets the MLflow experiment to "iris-flower-classification".
- Feature scaling is performed using `StandardScaler` and returned as pandas DataFrames to preserve feature names.

Dependencies:
- scikit-learn
- pandas
- MLflow
- logging

Run this script as a standalone module to execute the full training and registration workflow.
"""

# Import required libraries
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import logging
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set experiment in MLflow
mlflow.set_experiment("iris-flower-classification")


# Function to load and preprocess the data
def load_and_preprocess_data():
    try:
        # Load the Iris dataset
        iris = load_iris(as_frame=True)
        X = iris.frame.drop(columns="target")  # Features
        class_names = iris.target_names
        y = [
            class_names[i] for i in iris.target
        ]  # Convert numeric targets to string labels

        feature_columns = X.columns

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Standardize the features using StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_train = pd.DataFrame(X_train, columns=feature_columns)
        X_test = pd.DataFrame(X_test, columns=feature_columns)

        logger.info("Data loaded and preprocessed successfully.")
        return X_train, X_test, y_train, y_test

    except Exception as e:
        logger.error(f"Error loading or preprocessing data: {e}")
        raise


# Function to train the model, log metrics, and register the model
def train_and_log_model(X_train, y_train):
    try:
        # Initialize the classifier (Logistic Regression)
        model = LogisticRegression(max_iter=200)

        # Start an MLflow run
        with mlflow.start_run() as run:

            # Log parameters, metrics, and model automatically
            mlflow.sklearn.autolog(log_datasets=False)

            # Train the model on the training data
            model.fit(X_train, y_train)

            # Register the model explicitly
            model_uri = f"runs:/{run.info.run_id}/model"
            mlflow.register_model(model_uri=model_uri, name="iris-model")

            logger.info(f"Model registered successfully: iris-model")

    except Exception as e:
        logger.error(f"Error during model training or logging: {e}")
        raise


# Main function to orchestrate the workflow
def main():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    train_and_log_model(X_train, y_train)


if __name__ == "__main__":
    main()

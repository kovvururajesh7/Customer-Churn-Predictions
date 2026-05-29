# import joblib

# from sklearn.ensemble import RandomForestClassifier


# def train_model():
#     X_train = joblib.load("data/X_train.pkl")
#     y_train = joblib.load("data/y_train.pkl")

#     model = RandomForestClassifier(
#         n_estimators=100,
#         random_state=42
#     )

#     model.fit(X_train, y_train)

#     joblib.dump(model, "models/model.pkl")

#     print("Model trained successfully.")
#     print("Model saved at models/model.pkl")


# if __name__ == "__main__":
#     train_model()


import joblib
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train_model():
    
    # Load data
    X_train = joblib.load("data/X_train.pkl")
    X_test = joblib.load("data/X_test.pkl")

    y_train = joblib.load("data/y_train.pkl")
    y_test = joblib.load("data/y_test.pkl")

    # Create experiment
    mlflow.set_experiment("Customer_Churn_Prediction")

    with mlflow.start_run():

        # Hyperparameters
        n_estimators = 50
        random_state = 42

        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)

        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )

        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Metric
        accuracy = accuracy_score(y_test, y_pred)

        # Log metric
        mlflow.log_metric("accuracy", accuracy)

        # Save model artifact
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

        # Save local model
        joblib.dump(model, "models/model.pkl")

        print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    train_model()
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Remove rows with missing values
    df.dropna(inplace=True)

    # Convert categorical columns to numeric
    label_encoder = LabelEncoder()

    for column in df.select_dtypes(include="object").columns:
        if column != "Churn":
            df[column] = label_encoder.fit_transform(df[column])

    # Encode target column
    df["Churn"] = label_encoder.fit_transform(df["Churn"])

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    joblib.dump(X_train, "data/X_train.pkl")
    joblib.dump(X_test, "data/X_test.pkl")
    joblib.dump(y_train, "data/y_train.pkl")
    joblib.dump(y_test, "data/y_test.pkl")

    print("Preprocessing completed and files saved.")


if __name__ == "__main__":
    preprocess_data("data/customer_churn.csv")
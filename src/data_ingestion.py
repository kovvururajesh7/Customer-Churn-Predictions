import pandas as pd


def load_data(file_path):
    df = pd.read_csv(file_path)

    print(f"Dataset Shape: {df.shape}")
    print("\nMissing Values:")
    print(df.isnull().sum())

    return df


if __name__ == "__main__":
    df = load_data("data/customer_churn.csv")
    print(df.head())
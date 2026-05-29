from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict whether a customer will churn",
    version="1.0"
)

# Load trained model
model = joblib.load("models/model.pkl")


class CustomerData(BaseModel):
    customerID: int
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running"
    }


@app.post("/predict")
def predict(data: CustomerData):

    input_df = pd.DataFrame([data.model_dump()])

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    return {
        "prediction": int(prediction),
        "churn_probability": round(float(probability), 4)
    }
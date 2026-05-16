import pandas as pd
from fastapi import FastAPI

from app.schemas import MaintenanceInput
from app.model_loader import load_model


app = FastAPI(
    title="Maintenance Prediction API",
    description="API para predicción de fallas en mantenimiento predictivo",
    version="1.0.0"
)

model = load_model()


@app.get("/")
def root():
    return {
        "message": "Maintenance Prediction API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict(data: MaintenanceInput):
    input_data = pd.DataFrame([{
        "Type": data.Type,
        "Air temperature [K]": data.Air_temperature_K,
        "Process temperature [K]": data.Process_temperature_K,
        "Rotational speed [rpm]": data.Rotational_speed_rpm,
        "Torque [Nm]": data.Torque_Nm,
        "Tool wear [min]": data.Tool_wear_min
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0].max()

    return {
        "prediction": int(prediction),
        "failure_expected": bool(prediction),
        "probability": round(float(probability), 4)
    }
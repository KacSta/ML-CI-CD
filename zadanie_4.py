from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from sklearn.linear_model import LinearRegression
import os

app = FastAPI()

# odczyt zmiennych środowiskowych
API_KEY = os.getenv("API_KEY", "brak-klucza")
APP_ENV = os.getenv("APP_ENV", "development")

X = np.array([[10], [20], [30], [40], [50]])
y = np.array([21, 41, 61, 81, 101])

model = LinearRegression()
model.fit(X, y)

class PredictRequest(BaseModel):
    feature: float

@app.get("/")
def read_root():
    return {"message": "API ML gotowe do pracy. Sprawdź /docs po instrukcję."}

@app.get("/health")
def health_check():
    return {"status": "ok", "uptime": "stable"}

@app.get("/info")
def get_model_info():
    return {
        "model_type": type(model).__name__,
        "n_features_in": model.n_features_in_,
        "coefficients": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        "description": "Model regresji liniowej przewidujący y = 2x + 1"
    }

# endpoint pokazujący zmienne środowiskowe
@app.get("/config")
def get_config():
    return {
        "environment": APP_ENV,
        "api_key_set": API_KEY != "brak-klucza",
        "api_key_preview": f"{API_KEY[:4]}****" if len(API_KEY) > 4 else "brak-klucza"
    }

@app.post("/predict")
def predict(request: PredictRequest):
    if request.feature < 0:
        raise HTTPException(
            status_code=400,
            detail="Wartość cechy nie może być ujemna. Podaj liczbę dodatnią."
        )
    try:
        prediction = model.predict([[request.feature]])
        return {
            "input": request.feature,
            "prediction": float(prediction[0]),
            "environment": APP_ENV
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd modelu: {str(e)}")
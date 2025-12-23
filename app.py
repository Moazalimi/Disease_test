import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load trained model
with open("disease_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
columns = data["columns"]

app = FastAPI(title="Disease Prediction API")

class PredictRequest(BaseModel):
    symptoms: list[str]
    top_k: int = 5

@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/predict")
def predict(req: PredictRequest):
    input_vector = np.zeros(len(columns))

    for symptom in req.symptoms:
        if symptom in columns:
            input_vector[columns.index(symptom)] = 1

    input_df = pd.DataFrame([input_vector], columns=columns)

    probs = model.predict_proba(input_df)[0]
    diseases = model.classes_

    results = sorted(
        zip(diseases, probs),
        key=lambda x: x[1],
        reverse=True
    )

    return {
        "input_symptoms": req.symptoms,
        "predictions": [
            {
                "disease": d,
                "probability": round(float(p) * 100, 2)
            }
            for d, p in results[:req.top_k]
        ]
    }

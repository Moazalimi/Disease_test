import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load disease model (original) with pickle
with open(os.path.join(BASE_DIR, "disease_model.pkl"), "rb") as f:
    data = pickle.load(f)

model = data["model"]
columns = data["columns"]

# Load all trained specific disease models with joblib
specific_models = {}
model_files = {
    "heart": "heart_disease_rf_model.sav",
    "liver": "liver_disease_rf_model.sav",
    "lung": "lung_cancer_rf_model.sav",
    "parkinsons": "parkinsons_rf_model.sav"
}

for name, filename in model_files.items():
    filepath = os.path.join(BASE_DIR, filename)
    try:
        specific_models[name] = joblib.load(filepath)
        print(f"✓ Loaded {name} model from {filepath}")
    except FileNotFoundError:
        print(f"✗ Error: {filename} not found at {filepath}")
    except Exception as e:
        print(f"✗ Error loading {filename}: {e}")

app = FastAPI(title="Disease Prediction API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8081",
        "http://localhost:8080",
        "https://yourdomain.com"  # Replace with your actual deployed frontend URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Original request model
class PredictRequest(BaseModel):
    symptoms: list[str]
    top_k: int = 5

# Request models for specific diseases
class HeartDiseaseRequest(BaseModel):
    Age: int
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    MaxHR: int
    Oldpeak: float
    Sex_M: int
    ChestPainType_ATA: int
    ChestPainType_NAP: int
    ChestPainType_TA: int
    RestingECG_Normal: int
    RestingECG_ST: int
    ExerciseAngina_Y: int
    ST_Slope_Flat: int
    ST_Slope_Up: int

class LiverDiseaseRequest(BaseModel):
    Age: int
    Gender: int
    Total_Bilirubin: float
    Direct_Bilirubin: float
    Alkaline_Phosphotase: int
    Alamine_Aminotransferase: int
    Aspartate_Aminotransferase: int
    Total_Protiens: float
    Albumin: float
    Albumin_and_Globulin_Ratio: float

class LungCancerRequest(BaseModel):
    GENDER: int
    AGE: int
    SMOKING: int
    YELLOW_FINGERS: int
    ANXIETY: int
    PEER_PRESSURE: int
    CHRONIC_DISEASE: int
    FATIGUE: int
    ALLERGY: int
    WHEEZING: int
    ALCOHOL_CONSUMING: int
    COUGHING: int
    SHORTNESS_OF_BREATH: int
    SWALLOWING_DIFFICULTY: int
    CHEST_PAIN: int

class ParkinsonsRequest(BaseModel):
    MDVP_Fo_Hz: float
    MDVP_Fhi_Hz: float
    MDVP_Flo_Hz: float
    MDVP_Jitter_Percent: float
    MDVP_Jitter_Abs: float
    MDVP_RAP: float
    MDVP_PPQ: float
    Jitter_DDP: float
    MDVP_Shimmer: float
    MDVP_Shimmer_dB: float
    Shimmer_APQ3: float
    Shimmer_APQ5: float
    MDVP_APQ: float
    Shimmer_DDA: float
    NHR: float
    HNR: float
    RPDE: float
    DFA: float
    spread1: float
    spread2: float
    D2: float
    PPE: float

@app.get("/")
def health_check():
    return {"status": "API is running", "available_models": list(specific_models.keys())}

# Original disease prediction API
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

# Heart Disease prediction API
@app.post("/predict/heart")
def predict_heart(req: HeartDiseaseRequest):
    if "heart" not in specific_models:
        raise HTTPException(status_code=500, detail="Heart disease model not loaded")
    
    model_heart = specific_models["heart"]
    input_data = np.array([[
        req.Age, req.RestingBP, req.Cholesterol, req.FastingBS, req.MaxHR,
        req.Oldpeak, req.Sex_M, req.ChestPainType_ATA, req.ChestPainType_NAP,
        req.ChestPainType_TA, req.RestingECG_Normal, req.RestingECG_ST,
        req.ExerciseAngina_Y, req.ST_Slope_Flat, req.ST_Slope_Up
    ]])
    
    prediction = model_heart.predict(input_data)[0]
    probability = model_heart.predict_proba(input_data)[0]
    
    return {
        "disease": "Heart Disease",
        "prediction": int(prediction),
        "probability": {
            "class_0": round(float(probability[0]) * 100, 2),
            "class_1": round(float(probability[1]) * 100, 2)
        }
    }

# Liver Disease prediction API
@app.post("/predict/liver")
def predict_liver(req: LiverDiseaseRequest):
    if "liver" not in specific_models:
        raise HTTPException(status_code=500, detail="Liver disease model not loaded")
    
    model_liver = specific_models["liver"]
    input_data = np.array([[
        req.Age, req.Gender, req.Total_Bilirubin, req.Direct_Bilirubin,
        req.Alkaline_Phosphotase, req.Alamine_Aminotransferase,
        req.Aspartate_Aminotransferase, req.Total_Protiens, req.Albumin,
        req.Albumin_and_Globulin_Ratio
    ]])
    
    prediction = model_liver.predict(input_data)[0]
    probability = model_liver.predict_proba(input_data)[0]
    
    return {
        "disease": "Liver Disease",
        "prediction": int(prediction),
        "probability": {
            "class_0": round(float(probability[0]) * 100, 2),
            "class_1": round(float(probability[1]) * 100, 2)
        }
    }

# Lung Cancer prediction API
@app.post("/predict/lung")
def predict_lung(req: LungCancerRequest):
    if "lung" not in specific_models:
        raise HTTPException(status_code=500, detail="Lung cancer model not loaded")
    
    model_lung = specific_models["lung"]
    input_data = np.array([[
        req.GENDER, req.AGE, req.SMOKING, req.YELLOW_FINGERS, req.ANXIETY,
        req.PEER_PRESSURE, req.CHRONIC_DISEASE, req.FATIGUE, req.ALLERGY,
        req.WHEEZING, req.ALCOHOL_CONSUMING, req.COUGHING,
        req.SHORTNESS_OF_BREATH, req.SWALLOWING_DIFFICULTY, req.CHEST_PAIN
    ]])
    
    prediction = model_lung.predict(input_data)[0]
    probability = model_lung.predict_proba(input_data)[0]
    
    return {
        "disease": "Lung Cancer",
        "prediction": int(prediction),
        "probability": {
            "class_0": round(float(probability[0]) * 100, 2),
            "class_1": round(float(probability[1]) * 100, 2)
        }
    }

# Parkinson's Disease prediction API
@app.post("/predict/parkinsons")
def predict_parkinsons(req: ParkinsonsRequest):
    if "parkinsons" not in specific_models:
        raise HTTPException(status_code=500, detail="Parkinson's disease model not loaded")
    
    model_parkinsons = specific_models["parkinsons"]
    input_data = np.array([[
        req.MDVP_Fo_Hz, req.MDVP_Fhi_Hz, req.MDVP_Flo_Hz, req.MDVP_Jitter_Percent,
        req.MDVP_Jitter_Abs, req.MDVP_RAP, req.MDVP_PPQ, req.Jitter_DDP,
        req.MDVP_Shimmer, req.MDVP_Shimmer_dB, req.Shimmer_APQ3, req.Shimmer_APQ5,
        req.MDVP_APQ, req.Shimmer_DDA, req.NHR, req.HNR, req.RPDE, req.DFA,
        req.spread1, req.spread2, req.D2, req.PPE
    ]])
    
    prediction = model_parkinsons.predict(input_data)[0]
    probability = model_parkinsons.predict_proba(input_data)[0]
    
    return {
        "disease": "Parkinson's Disease",
        "prediction": int(prediction),
        "probability": {
            "class_0": round(float(probability[0]) * 100, 2),
            "class_1": round(float(probability[1]) * 100, 2)
        }
    }

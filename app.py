import pickle
import numpy as np
import pandas as pd
import logging
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Disease Prediction APIs")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model paths - use environment variable or default
MODEL_DIR = os.getenv("MODEL_DIR", "models")

# Load all models
models = {}

model_files = {
    "heart": "heart_disease_rf_model.sav",
    "liver": "liver_disease_rf_model.sav",
    "lung": "lung_cancer_rf_model.sav",
    "parkinsons": "parkinsons_rf_model.sav"
}

for key, filename in model_files.items():
    try:
        path = os.path.join(MODEL_DIR, filename)
        with open(path, "rb") as f:
            models[key] = pickle.load(f)
            logger.info(f"Loaded {key} model successfully")
    except Exception as e:
        logger.error(f"Error loading {key} model: {e}")

disease_model_data = None
try:
    path = os.path.join(MODEL_DIR, "disease_model.pkl")
    with open(path, "rb") as f:
        disease_model_data = pickle.load(f)
        logger.info("Loaded disease model successfully")
except Exception as e:
    logger.error(f"Error loading disease model: {e}")

# Base request models
class HealthCheckResponse(BaseModel):
    status: str
    available_models: List[str]

class PredictionResponse(BaseModel):
    disease: str
    probability: float

class SymptomPredictRequest(BaseModel):
    symptoms: List[str]
    top_k: int = Field(default=5, ge=1, le=20)

class HeartDiseaseRequest(BaseModel):
    age: int = Field(ge=0, le=150)
    resting_bp: int = Field(ge=0, le=300)
    cholesterol: int = Field(ge=0, le=600)
    fasting_bs: int = Field(ge=0, le=1)
    max_hr: int = Field(ge=0, le=300)
    oldpeak: float = Field(ge=0, le=10)
    sex_m: int = Field(ge=0, le=1)
    chest_pain_type_ata: int = Field(ge=0, le=1)
    chest_pain_type_nap: int = Field(ge=0, le=1)
    chest_pain_type_ta: int = Field(ge=0, le=1)
    resting_ecg_normal: int = Field(ge=0, le=1)
    resting_ecg_st: int = Field(ge=0, le=1)
    exercise_angina_y: int = Field(ge=0, le=1)
    st_slope_flat: int = Field(ge=0, le=1)
    st_slope_up: int = Field(ge=0, le=1)

class LiverDiseaseRequest(BaseModel):
    age: int = Field(ge=0, le=150)
    gender: int = Field(ge=0, le=1)
    total_bilirubin: float = Field(ge=0)
    direct_bilirubin: float = Field(ge=0)
    alkaline_phosphatase: int = Field(ge=0)
    alamine_aminotransferase: int = Field(ge=0)
    aspartate_aminotransferase: int = Field(ge=0)
    total_proteins: float = Field(ge=0)
    albumin: float = Field(ge=0)
    albumin_and_globulin_ratio: float = Field(ge=0)

class LungCancerRequest(BaseModel):
    gender: int = Field(ge=0, le=1)
    age: int = Field(ge=0, le=150)
    smoking: int = Field(ge=0, le=1)
    yellow_fingers: int = Field(ge=0, le=1)
    anxiety: int = Field(ge=0, le=1)
    peer_pressure: int = Field(ge=0, le=1)
    chronic_disease: int = Field(ge=0, le=1)
    fatigue: int = Field(ge=0, le=1)
    allergy: int = Field(ge=0, le=1)
    wheezing: int = Field(ge=0, le=1)
    alcohol_consuming: int = Field(ge=0, le=1)
    coughing: int = Field(ge=0, le=1)
    shortness_of_breath: int = Field(ge=0, le=1)
    swallowing_difficulty: int = Field(ge=0, le=1)
    chest_pain: int = Field(ge=0, le=1)

class ParkinsonsRequest(BaseModel):
    MDVP_Fo_Hz: float = Field(ge=0)
    MDVP_Fhi_Hz: float = Field(ge=0)
    MDVP_Flo_Hz: float = Field(ge=0)
    MDVP_Jitter_Percent: float = Field(ge=0)
    MDVP_Jitter_Abs: float = Field(ge=0)
    MDVP_RAP: float = Field(ge=0)
    MDVP_PPQ: float = Field(ge=0)
    Jitter_DDP: float = Field(ge=0)
    MDVP_Shimmer: float = Field(ge=0)
    MDVP_Shimmer_dB: float = Field(ge=0)
    Shimmer_APQ3: float = Field(ge=0)
    Shimmer_APQ5: float = Field(ge=0)
    MDVP_APQ: float = Field(ge=0)
    Shimmer_DDA: float = Field(ge=0)
    NHR: float = Field(ge=0)
    HNR: float = Field(ge=0)
    RPDE: float = Field(ge=0)
    DFA: float = Field(ge=0)
    spread1: float
    spread2: float
    D2: float = Field(ge=0)
    PPE: float = Field(ge=0)

@app.get("/", response_model=HealthCheckResponse)
def health_check():
    """Check API health and available models"""
    available = list(models.keys())
    if disease_model_data is not None:
        available.append("symptom-based")
    return {
        "status": "API is running",
        "available_models": available
    }

@app.post("/predict/symptoms")
def predict_symptoms(req: SymptomPredictRequest) -> Dict[str, Any]:
    """Predict diseases based on symptoms"""
    if disease_model_data is None:
        raise HTTPException(status_code=503, detail="Disease model not loaded")
    
    try:
        model = disease_model_data["model"]
        columns = disease_model_data["columns"]
        
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
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.post("/predict/heart")
def predict_heart_disease(req: HeartDiseaseRequest) -> Dict[str, Any]:
    """Predict heart disease probability"""
    if "heart" not in models:
        raise HTTPException(status_code=503, detail="Heart disease model not loaded")
    
    try:
        model = models["heart"]
        input_data = np.array([
            [
                req.age, req.resting_bp, req.cholesterol, req.fasting_bs,
                req.max_hr, req.oldpeak, req.sex_m, req.chest_pain_type_ata,
                req.chest_pain_type_nap, req.chest_pain_type_ta, req.resting_ecg_normal,
                req.resting_ecg_st, req.exercise_angina_y, req.st_slope_flat,
                req.st_slope_up
            ]
        ])
        
        prediction = model.predict(input_data)[0]
        probability = max(model.predict_proba(input_data)[0]) * 100
        
        return {
            "disease": "Heart Disease",
            "prediction": int(prediction),
            "probability": round(float(probability), 2),
            "risk_level": "High" if prediction == 1 else "Low"
        }
    except Exception as e:
        logger.error(f"Heart prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.post("/predict/liver")
def predict_liver_disease(req: LiverDiseaseRequest) -> Dict[str, Any]:
    """Predict liver disease probability"""
    if "liver" not in models:
        raise HTTPException(status_code=503, detail="Liver disease model not loaded")
    
    try:
        model = models["liver"]
        input_data = np.array([
            [
                req.age, req.gender, req.total_bilirubin, req.direct_bilirubin,
                req.alkaline_phosphatase, req.alamine_aminotransferase,
                req.aspartate_aminotransferase, req.total_proteins, req.albumin,
                req.albumin_and_globulin_ratio
            ]
        ])
        
        prediction = model.predict(input_data)[0]
        probability = max(model.predict_proba(input_data)[0]) * 100
        
        return {
            "disease": "Liver Disease",
            "prediction": int(prediction),
            "probability": round(float(probability), 2),
            "risk_level": "High" if prediction == 1 else "Low"
        }
    except Exception as e:
        logger.error(f"Liver prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.post("/predict/lung")
def predict_lung_cancer(req: LungCancerRequest) -> Dict[str, Any]:
    """Predict lung cancer probability"""
    if "lung" not in models:
        raise HTTPException(status_code=503, detail="Lung cancer model not loaded")
    
    try:
        model = models["lung"]
        input_data = np.array([
            [
                req.gender, req.age, req.smoking, req.yellow_fingers,
                req.anxiety, req.peer_pressure, req.chronic_disease,
                req.fatigue, req.allergy, req.wheezing, req.alcohol_consuming,
                req.coughing, req.shortness_of_breath, req.swallowing_difficulty,
                req.chest_pain
            ]
        ])
        
        prediction = model.predict(input_data)[0]
        probability = max(model.predict_proba(input_data)[0]) * 100
        
        return {
            "disease": "Lung Cancer",
            "prediction": int(prediction),
            "probability": round(float(probability), 2),
            "risk_level": "High" if prediction == 1 else "Low"
        }
    except Exception as e:
        logger.error(f"Lung prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.post("/predict/parkinsons")
def predict_parkinsons(req: ParkinsonsRequest) -> Dict[str, Any]:
    """Predict Parkinson's disease probability"""
    if "parkinsons" not in models:
        raise HTTPException(status_code=503, detail="Parkinson's model not loaded")
    
    try:
        model = models["parkinsons"]
        input_data = np.array([
            [
                req.MDVP_Fo_Hz, req.MDVP_Fhi_Hz, req.MDVP_Flo_Hz,
                req.MDVP_Jitter_Percent, req.MDVP_Jitter_Abs, req.MDVP_RAP,
                req.MDVP_PPQ, req.Jitter_DDP, req.MDVP_Shimmer,
                req.MDVP_Shimmer_dB, req.Shimmer_APQ3, req.Shimmer_APQ5,
                req.MDVP_APQ, req.Shimmer_DDA, req.NHR, req.HNR,
                req.RPDE, req.DFA, req.spread1, req.spread2,
                req.D2, req.PPE
            ]
        ])
        
        prediction = model.predict(input_data)[0]
        probability = max(model.predict_proba(input_data)[0]) * 100
        
        return {
            "disease": "Parkinson's Disease",
            "prediction": int(prediction),
            "probability": round(float(probability), 2),
            "risk_level": "High" if prediction == 1 else "Low"
        }
    except Exception as e:
        logger.error(f"Parkinson's prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

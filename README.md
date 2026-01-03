# Disease Prediction API

A FastAPI-based REST API for predicting multiple diseases using machine learning models. This application provides endpoints for predicting Heart Disease, Liver Disease, Lung Cancer, and Parkinson's Disease.

## Features

- Multiple disease prediction models
- RESTful API endpoints
- Interactive API documentation (Swagger UI)
- Real-time predictions with probability scores
- General symptom-based disease prediction

## Technologies Used

- **Framework**: FastAPI
- **Server**: Uvicorn
- **ML Library**: scikit-learn
- **Data Processing**: pandas, numpy
- **Model Serialization**: joblib
- **Python Version**: 3.12

## Installation

1. Clone the repository or navigate to the project directory

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

The server will run on `http://127.0.0.1:8000`

## API Documentation

### Interactive Documentation

Once the server is running, access the interactive API documentation at:

- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

## Endpoints

### 1. Health Check

**GET** `/`

Check if API is running and view available models.

**Response**:
```json
{
  "status": "API is running",
  "available_models": ["heart", "liver", "lung", "parkinsons"]
}
```

---

### 2. General Disease Prediction

**POST** `/predict`

Predict diseases based on symptoms.

**Request**:
```json
{
  "symptoms": ["fever", "cough", "fatigue"],
  "top_k": 5
}
```

**Response**:
```json
{
  "input_symptoms": ["fever", "cough", "fatigue"],
  "predictions": [
    {
      "disease": "Common Cold",
      "probability": 45.23
    },
    {
      "disease": "Flu",
      "probability": 32.15
    }
  ]
}
```

---

### 3. Heart Disease Prediction

**POST** `/predict/heart`

Predict heart disease risk using 15 clinical parameters.

**Request**:
```json
{
  "Age": 45,
  "RestingBP": 120,
  "Cholesterol": 200,
  "FastingBS": 0,
  "MaxHR": 150,
  "Oldpeak": 1.5,
  "Sex_M": 1,
  "ChestPainType_ATA": 1,
  "ChestPainType_NAP": 0,
  "ChestPainType_TA": 0,
  "RestingECG_Normal": 1,
  "RestingECG_ST": 0,
  "ExerciseAngina_Y": 0,
  "ST_Slope_Flat": 0,
  "ST_Slope_Up": 1
}
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| Age | int | Years |
| RestingBP | int | mm Hg |
| Cholesterol | int | mg/dL |
| FastingBS | int | 1 if >120 mg/dL else 0 |
| MaxHR | int | Maximum heart rate |
| Oldpeak | float | ST depression |
| Sex_M | int | 1 = Male, 0 = Female |
| ChestPainType_ATA | int | One-hot encoded |
| ChestPainType_NAP | int | One-hot encoded |
| ChestPainType_TA | int | One-hot encoded |
| RestingECG_Normal | int | One-hot encoded |
| RestingECG_ST | int | One-hot encoded |
| ExerciseAngina_Y | int | 1 = Yes, 0 = No |
| ST_Slope_Flat | int | One-hot encoded |
| ST_Slope_Up | int | One-hot encoded |

**Response**:
```json
{
  "disease": "Heart Disease",
  "prediction": 0,
  "probability": {
    "class_0": 75.45,
    "class_1": 24.55
  }
}
```

---

### 4. Liver Disease Prediction

**POST** `/predict/liver`

Predict liver disease risk using 10 laboratory parameters.

**Request**:
```json
{
  "Age": 50,
  "Gender": 1,
  "Total_Bilirubin": 0.8,
  "Direct_Bilirubin": 0.2,
  "Alkaline_Phosphotase": 60,
  "Alamine_Aminotransferase": 28,
  "Aspartate_Aminotransferase": 35,
  "Total_Protiens": 7.0,
  "Albumin": 3.5,
  "Albumin_and_Globulin_Ratio": 1.2
}
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| Age | int | Years |
| Gender | int | 1 = Male, 0 = Female |
| Total_Bilirubin | float | mg/dL |
| Direct_Bilirubin | float | mg/dL |
| Alkaline_Phosphotase | int | IU/L |
| Alamine_Aminotransferase | int | IU/L |
| Aspartate_Aminotransferase | int | IU/L |
| Total_Protiens | float | g/dL |
| Albumin | float | g/dL |
| Albumin_and_Globulin_Ratio | float | Ratio |

**Response**:
```json
{
  "disease": "Liver Disease",
  "prediction": 0,
  "probability": {
    "class_0": 68.32,
    "class_1": 31.68
  }
}
```

---

### 5. Lung Cancer Prediction

**POST** `/predict/lung`

Predict lung cancer risk using 15 binary survey parameters.

**Request**:
```json
{
  "GENDER": 1,
  "AGE": 55,
  "SMOKING": 1,
  "YELLOW_FINGERS": 0,
  "ANXIETY": 0,
  "PEER_PRESSURE": 0,
  "CHRONIC_DISEASE": 0,
  "FATIGUE": 1,
  "ALLERGY": 0,
  "WHEEZING": 1,
  "ALCOHOL_CONSUMING": 0,
  "COUGHING": 1,
  "SHORTNESS_OF_BREATH": 1,
  "SWALLOWING_DIFFICULTY": 0,
  "CHEST_PAIN": 0
}
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| GENDER | int | 1 = Male, 0 = Female |
| AGE | int | Years |
| SMOKING | int | 1 = Yes, 0 = No |
| YELLOW_FINGERS | int | 1 = Yes, 0 = No |
| ANXIETY | int | 1 = Yes, 0 = No |
| PEER_PRESSURE | int | 1 = Yes, 0 = No |
| CHRONIC_DISEASE | int | 1 = Yes, 0 = No |
| FATIGUE | int | 1 = Yes, 0 = No |
| ALLERGY | int | 1 = Yes, 0 = No |
| WHEEZING | int | 1 = Yes, 0 = No |
| ALCOHOL_CONSUMING | int | 1 = Yes, 0 = No |
| COUGHING | int | 1 = Yes, 0 = No |
| SHORTNESS_OF_BREATH | int | 1 = Yes, 0 = No |
| SWALLOWING_DIFFICULTY | int | 1 = Yes, 0 = No |
| CHEST_PAIN | int | 1 = Yes, 0 = No |

**Response**:
```json
{
  "disease": "Lung Cancer",
  "prediction": 1,
  "probability": {
    "class_0": 35.67,
    "class_1": 64.33
  }
}
```

---

### 6. Parkinson's Disease Prediction

**POST** `/predict/parkinsons`

Predict Parkinson's disease risk using 22 voice feature parameters.

**Request**:
```json
{
  "MDVP_Fo_Hz": 120.5,
  "MDVP_Fhi_Hz": 130.2,
  "MDVP_Flo_Hz": 110.8,
  "MDVP_Jitter_Percent": 0.005,
  "MDVP_Jitter_Abs": 0.00005,
  "MDVP_RAP": 0.003,
  "MDVP_PPQ": 0.004,
  "Jitter_DDP": 0.009,
  "MDVP_Shimmer": 0.03,
  "MDVP_Shimmer_dB": 0.3,
  "Shimmer_APQ3": 0.015,
  "Shimmer_APQ5": 0.02,
  "MDVP_APQ": 0.025,
  "Shimmer_DDA": 0.04,
  "NHR": 0.01,
  "HNR": 30.0,
  "RPDE": 0.5,
  "DFA": 0.65,
  "spread1": -5.5,
  "spread2": 0.2,
  "D2": 2.5,
  "PPE": 0.1
}
```

**Parameters**:
| Parameter | Type |
|-----------|------|
| MDVP_Fo_Hz | float |
| MDVP_Fhi_Hz | float |
| MDVP_Flo_Hz | float |
| MDVP_Jitter_Percent | float |
| MDVP_Jitter_Abs | float |
| MDVP_RAP | float |
| MDVP_PPQ | float |
| Jitter_DDP | float |
| MDVP_Shimmer | float |
| MDVP_Shimmer_dB | float |
| Shimmer_APQ3 | float |
| Shimmer_APQ5 | float |
| MDVP_APQ | float |
| Shimmer_DDA | float |
| NHR | float |
| HNR | float |
| RPDE | float |
| DFA | float |
| spread1 | float |
| spread2 | float |
| D2 | float |
| PPE | float |

**Response**:
```json
{
  "disease": "Parkinson's Disease",
  "prediction": 0,
  "probability": {
    "class_0": 82.45,
    "class_1": 17.55
  }
}
```

---

## Response Format

All disease-specific endpoints return a JSON response with:

- **disease** (string): Name of the disease being predicted
- **prediction** (integer): 0 = Negative (no disease), 1 = Positive (disease detected)
- **probability** (object):
  - **class_0** (float): Probability of no disease (%)
  - **class_1** (float): Probability of disease present (%)

## Error Handling

If a model fails to load, the API will return a 500 error:

```json
{
  "detail": "Model name not loaded"
}
```

## Required Files

Ensure the following model files are in the same directory as `app.py`:

- `disease_model.pkl` - General disease prediction model
- `heart_disease_rf_model.sav` - Heart disease model
- `liver_disease_rf_model.sav` - Liver disease model
- `lung_cancer_rf_model.sav` - Lung cancer model
- `parkinsons_rf_model.sav` - Parkinson's disease model

## Project Structure

```
Disease_test/
├── app.py
├── requirements.txt
├── README.md
├── disease_model.pkl
├── heart_disease_rf_model.sav
├── liver_disease_rf_model.sav
├── lung_cancer_rf_model.sav
└── parkinsons_rf_model.sav
```

## Testing the API

### Using Swagger UI (Recommended)

1. Start the server: `uvicorn app:app --reload`
2. Open browser: http://127.0.0.1:8000/docs
3. Click on any endpoint
4. Click "Try it out"
5. Enter test values and click "Execute"

### Using cURL

```bash
# Test heart disease prediction
curl -X POST "http://127.0.0.1:8000/predict/heart" \
  -H "Content-Type: application/json" \
  -d '{"Age": 45, "RestingBP": 120, "Cholesterol": 200, "FastingBS": 0, "MaxHR": 150, "Oldpeak": 1.5, "Sex_M": 1, "ChestPainType_ATA": 1, "ChestPainType_NAP": 0, "ChestPainType_TA": 0, "RestingECG_Normal": 1, "RestingECG_ST": 0, "ExerciseAngina_Y": 0, "ST_Slope_Flat": 0, "ST_Slope_Up": 1}'
```

## License

This project is for educational purposes.

## Author

Moaz Alimi - 8th Semester Project
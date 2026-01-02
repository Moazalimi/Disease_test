# Disease Prediction API

A FastAPI application for predicting various diseases using machine learning models.

## Supported Diseases
- Heart Disease
- Liver Disease
- Lung Cancer
- Parkinson's Disease
- Symptom-based Disease Prediction

## Setup

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Add model files to `models/` directory:
- `heart_disease_rf_model.sav`
- `liver_disease_rf_model.sav`
- `lung_cancer_rf_model.sav`
- `parkinsons_rf_model.sav`
- `disease_model.pkl`

## Running Locally

```bash
uvicorn app:app --reload
```

API available at `http://localhost:8000`

## Deployment on Render

1. Push to GitHub
2. Create new Web Service on Render
3. Connect GitHub repository
4. Set start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Add environment variable: `MODEL_DIR=models`
6. Deploy

## API Endpoints

- `GET /` - Health check
- `POST /predict/heart` - Heart disease prediction
- `POST /predict/liver` - Liver disease prediction
- `POST /predict/lung` - Lung cancer prediction
- `POST /predict/parkinsons` - Parkinson's disease prediction
- `POST /predict/symptoms` - Symptom-based prediction

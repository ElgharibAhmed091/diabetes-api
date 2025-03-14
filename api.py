from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load saved artifacts
model = joblib.load('diabetes_model.pkl')
encoder = joblib.load('label_encoder.pkl')

class PatientData(BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    smoking_history: str
    bmi: float
    HbA1c_level: float
    blood_glucose_level: int

@app.post("/predict")
async def predict_diabetes(data: PatientData):
    try:
        input_data = {
            'gender': data.gender,
            'age': data.age,
            'hypertension': data.hypertension,
            'heart_disease': data.heart_disease,
            'smoking_history': data.smoking_history,
            'bmi': data.bmi,
            'HbA1c_level': data.HbA1c_level,
            'blood_glucose_level': data.blood_glucose_level
        }
        
        input_data['gender'] = encoder.transform([input_data['gender']])[0]
        input_data['smoking_history'] = encoder.transform([input_data['smoking_history']])[0]
        
        features = np.array(list(input_data.values())).reshape(1, -1)
        
        prediction = model.predict(features)
        
        return {"prediction": int(prediction[0])}
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
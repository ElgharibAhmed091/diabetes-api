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

        # Encode categorical data
        input_data['gender'] = encoder.transform([input_data['gender']])[0]
        input_data['smoking_history'] = encoder.transform([input_data['smoking_history']])[0]

        # Prepare features for prediction
        features = np.array(list(input_data.values())).reshape(1, -1)

        # Predict probability of diabetes (as a percentage)
        probability = model.predict_proba(features)[0][1] * 100  # index 1 is the positive class

        # Recommendations based on probability
        if probability < 50:
            recommendation = {
                "risk_level": "Mild Risk 0-50%",
                "message": "Your symptoms show a low risk of Diabetes. Stay aware and maintain healthy habits.",
                "actions": [
                    "Keep an eye on your health.",
                    "Follow a balanced diet.",
                    "Get routine check-up.",
                    "Stay active & Exercise regularly."
                ]
            }
        else:
            recommendation = {
                "risk_level": "Severe Risk 50-100%",
                "message": "Your symptoms suggest a moderate to high risk of Diabetes. Seek medical help ASAP.",
                "actions": [
                    "Contact a healthcare professional immediately.",
                    "Visit a Clinic or Hospital if Symptoms Worsen.",
                    "Follow Medical Treatment & Recommendations.",
                    "Get tested to confirm diagnosis."
                ]
            }

        return {
            "prediction_percentage": round(probability, 2),
            "recommendation": recommendation
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

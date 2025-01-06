from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# Initialize app
app = FastAPI()

## upload the model
model=pickle.load(open('model.pkl','rb'))

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],  # Adjust this to your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class WeatherData(BaseModel):
    temperature: float
    humidity: float
    wind_speed: float
    cloud_cover: float
    pressure: float

@app.post("/predict/")
def predict_rain(data: WeatherData):
    try:
        # Prepare data for prediction
        features = np.array([[data.temperature, data.humidity, data.wind_speed, data.cloud_cover, data.pressure]])
        
        # Predict using the model
        prediction = model.predict(features)
        
        if prediction[0]==0:
            return {'prediction':'No rain'}
        else :
            return {'prediction':'Rain'}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

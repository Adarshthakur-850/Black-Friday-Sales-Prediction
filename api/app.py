from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import pandas as pd
import numpy as np
from src.inference import SalesPredictor
from src.utils import logger
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Black Friday Sales Prediction API", version="1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Predictor
predictor = SalesPredictor()

class PredictionInput(BaseModel):
    User_ID: int
    Product_ID: str
    Gender: str
    Age: str
    Occupation: int
    City_Category: str
    Stay_In_Current_City_Years: str
    Marital_Status: int
    Product_Category_1: int
    Product_Category_2: float = 0.0
    Product_Category_3: float = 0.0

@app.get("/")
def home():
    return {"message": "Welcome to Black Friday Sales Prediction API"}

@app.post("/predict")
def predict(input_data: PredictionInput):
    """
    Predicts purchase amount.
    """
    try:
        data = input_data.dict()
        prediction = predictor.predict(data)
        return {
            "prediction": float(prediction[0]),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

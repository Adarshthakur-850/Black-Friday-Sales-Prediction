import pandas as pd
import numpy as np
from src.utils import load_object, logger
from src.config import BEST_MODEL_PATH

class SalesPredictor:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            self.model = load_object(BEST_MODEL_PATH)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None

    def predict(self, data: dict):
        """
        Predicts purchase amount for a single input (dict) or list of inputs.
        """
        if self.model is None:
            raise ValueError("Model is not loaded.")
        
        # Convert dictionary to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            raise ValueError("Input data must be a dictionary or list of dictionaries.")
        
        try:
            prediction = self.model.predict(df)
            return prediction
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise

if __name__ == "__main__":
    # Test inference
    predictor = SalesPredictor()
    # Dummy sample matching schema
    sample = {
        'User_ID': 1000001,
        'Product_ID': 'P00069042',
        'Gender': 'F',
        'Age': '0-17',
        'Occupation': 10,
        'City_Category': 'A',
        'Stay_In_Current_City_Years': '2',
        'Marital_Status': 0,
        'Product_Category_1': 3,
        'Product_Category_2': 5.0,
        'Product_Category_3': np.nan 
    }
    
    try:
        pred = predictor.predict(sample)
        print(f"Predicted Purchase: {pred[0]}")
    except Exception as e:
        print(e)

import pandas as pd
import numpy as np
from pathlib import Path
from src.config import DATA_DIR, TRAIN_DATA_PATH
from src.utils import logger

def generate_synthetic_data(num_samples=10000):
    """
    Generates a synthetic Black Friday dataset similar to the original.
    """
    logger.info(f"Generating synthetic data with {num_samples} samples...")
    
    np.random.seed(42)
    
    data = {
        'User_ID': np.random.randint(1000001, 1005000, num_samples),
        'Product_ID': [f'P00{np.random.randint(100, 999)}42' for _ in range(num_samples)],
        'Gender': np.random.choice(['M', 'F'], num_samples, p=[0.75, 0.25]),
        'Age': np.random.choice(['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+'], num_samples),
        'Occupation': np.random.randint(0, 21, num_samples),
        'City_Category': np.random.choice(['A', 'B', 'C'], num_samples),
        'Stay_In_Current_City_Years': np.random.choice(['0', '1', '2', '3', '4+'], num_samples),
        'Marital_Status': np.random.choice([0, 1], num_samples),
        'Product_Category_1': np.random.randint(1, 19, num_samples),
        'Product_Category_2': np.random.choice(list(range(2, 19)) + [np.nan], num_samples, p=[0.05]*17 + [0.15]),
        'Product_Category_3': np.random.choice(list(range(3, 19)) + [np.nan], num_samples, p=[0.04]*16 + [0.36]),
        'Purchase': np.random.randint(100, 20000, num_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TRAIN_DATA_PATH, index=False)
    logger.info(f"Synthetic data saved to {TRAIN_DATA_PATH}")

if __name__ == "__main__":
    generate_synthetic_data()

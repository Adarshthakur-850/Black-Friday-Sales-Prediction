import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.config import TRAIN_DATA_PATH, BEST_MODEL_PATH, RANDOM_STATE, TEST_SIZE
from src.utils import logger, save_object
from src.preprocessing import Preprocessor
from src.feature_engineering import FeatureEngineer

def train_model():
    logger.info("Loading data...")
    try:
        df = pd.read_csv(TRAIN_DATA_PATH)
    except FileNotFoundError:
        logger.error(f"Data file not found at {TRAIN_DATA_PATH}. Please run generate_data.py first.")
        return

    # Separate target
    X = df.drop('Purchase', axis=1)
    y = df['Purchase']

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Define Models
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, n_jobs=-1, random_state=RANDOM_STATE)
    }

    best_model = None
    best_rmse = float('inf')
    results = {}

    logger.info("Starting model comparison...")
    
    for name, model in models.items():
        # Create Pipeline
        pipeline = Pipeline([
            ('preprocessing', Preprocessor()),
            ('feature_engineering', FeatureEngineer(add_kmeans=True)), # Feature Engineering after Preprocessing? 
            # Wait, Feature Engineering adds columns like 'Customer_Segment' using KMeans on raw/processed data?
            # My FeatureEngineer expects raw DataFrame for some cols.
            # Preprocessor returns numpy array (ColumnTransformer).
            # If Preprocessor runs first, FeatureEngineer will receive numpy array and lose column names.
            # We should swap or adjust.
            # Actually, `FeatureEngineer` in my implementation uses column names like 'Product_Category_1'.
            # So it must run BEFORE Preprocessor if Preprocessor outputs numpy array.
            # But Preprocessor handles OneHot encoding which might be needed for model.
            
            # Let's adjust: FeatureEngineer -> Preprocessor -> Model
            # FeatureEngineer takes DF, adds columns, returns DF.
            # Preprocessor takes DF, encoding/scaling, returns Array.
            # Model takes Array.
            
            # Correction: My FeatureEngineer implementation returns a DataFrame (X_copy).
            # So Pipeline: FeatureEngineer -> Preprocessor -> Model is correct.
        ])
        
        # Redefine pipeline structure
        pipeline = Pipeline([
            ('feature_engineering', FeatureEngineer(add_kmeans=True)),
            ('preprocessing', Preprocessor()),
            ('model', model)
        ])

        # Train
        logger.info(f"Training {name}...")
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = pipeline.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}
        logger.info(f"{name} Results - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = pipeline

    logger.info("--------------------------------------------------")
    logger.info(f"Best Model: {best_model.steps[-1][1].__class__.__name__} with RMSE: {best_rmse}")
    
    # Save Best Model
    save_object(best_model, BEST_MODEL_PATH)
    
    # Save results (optional, maybe to a file)
    return results

if __name__ == "__main__":
    train_model()

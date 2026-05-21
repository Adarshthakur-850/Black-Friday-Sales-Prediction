import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.utils import logger, save_object
from src.config import ENCODER_PATH, SCALER_PATH

class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = None
        self.scaler = None
        self.cat_cols = ['Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years', 'City_Occupation', 'Customer_Segment']
        self.num_cols = ['Occupation', 'Marital_Status', 'Product_Category_1', 'Product_Category_2', 'Product_Category_3']
        # Note: Handling Purchase is done separately (target)
        
        # Pipeline for categorical features
        self.cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')), # Should wrap if needed, but for OneHot we deal below
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Pipeline for numerical features
        self.num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)), # Replace NaNs with 0 for categories
            ('scaler', StandardScaler())
        ])
        
        self.column_transformer = None

    def fit(self, X, y=None):
        logger.info("Fitting preprocessor...")
        
        # Define high cardinality vs low cardinality handling if needed
        # For this dataset, standard OneHot for low card and handling numericals
        
        self.column_transformer = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.cat_cols),
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                    ('scaler', StandardScaler())
                ]), self.num_cols)
            ]
        )
        
        self.column_transformer.fit(X)
        
        # Save artifacts
        save_object(self.column_transformer, ENCODER_PATH) # Saving the whole transformer as encoder for simplicity
        
        return self

    def transform(self, X):
        logger.info("Transforming data...")
        if self.column_transformer is None:
             raise ValueError("Preprocessor not fitted.")
        return self.column_transformer.transform(X)

# We can also use functional approach or simpler pipeline construction
from sklearn.impute import SimpleImputer

def get_preprocessor_pipeline():
    """
    Returns a ColumnTransformer pipeline.
    """
    cat_cols = ['Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years']
    num_cols = ['Occupation', 'Marital_Status', 'Product_Category_1', 'Product_Category_2', 'Product_Category_3']
    
    # Interactions and Feature Engineering are passed in as dataframe, so preprocess raw columns
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ])

    return preprocessor

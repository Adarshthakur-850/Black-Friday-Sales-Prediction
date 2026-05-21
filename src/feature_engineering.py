import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from src.utils import logger

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, add_kmeans=True, n_clusters=3):
        self.add_kmeans = add_kmeans
        self.n_clusters = n_clusters
        self.kmeans = None

    def fit(self, X, y=None):
        if self.add_kmeans:
            # Fit KMeans on a subset of features (e.g., Product Categories or Demographics)
            # For simplicity, let's use Product_Category_1 and Occupation
            logger.info("Fitting KMeans for customer segmentation...")
            subset = X[['Product_Category_1', 'Occupation']].fillna(0)
            self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            self.kmeans.fit(subset)
        return self

    def transform(self, X):
        logger.info("Performing feature engineering...")
        X_copy = X.copy()
        
        # 1. City_Occupation Interaction
        X_copy['City_Occupation'] = X_copy['City_Category'].astype(str) + "_" + X_copy['Occupation'].astype(str)
        
        # 2. Product_Category_2 and 3 filling (handled in robust preprocessing usually, but explicitly here for features)
        X_copy['Product_Category_2'] = X_copy['Product_Category_2'].fillna(0)
        X_copy['Product_Category_3'] = X_copy['Product_Category_3'].fillna(0)
        
        # 3. Age Group Mapping (Logic often done here if not raw)
        # Age is already categorical 0-17, etc. in original dataset
        
        # 4. KMeans Cluster
        if self.add_kmeans and self.kmeans:
            subset = X_copy[['Product_Category_1', 'Occupation']]
            X_copy['Customer_Segment'] = self.kmeans.predict(subset)
            
        return X_copy

def feature_engineering_pipeline(df: pd.DataFrame, train=True):
    """
    Functional wrapper if class not used directly.
    """
    fe = FeatureEngineer()
    if train:
        fe.fit(df)
    return fe.transform(df)

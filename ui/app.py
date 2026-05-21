import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

st.set_page_config(page_title="Black Friday Sales Prediction", layout="wide")

st.title("🛍️ Black Friday Sales Prediction")
st.markdown("Predict purchase amount based on customer demographics and product details.")

# Sidebar for Input
st.sidebar.header("Customer & Product Details")

with st.sidebar.form("prediction_form"):
    user_id = st.number_input("User ID", min_value=1000000, value=1000001)
    product_id = st.text_input("Product ID", value="P00069042")
    
    gender = st.selectbox("Gender", ["M", "F"])
    age = st.selectbox("Age Group", ["0-17", "18-25", "26-35", "36-45", "46-50", "51-55", "55+"])
    occupation = st.number_input("Occupation (0-20)", min_value=0, max_value=20, value=10)
    city_category = st.selectbox("City Category", ["A", "B", "C"])
    stay_years = st.selectbox("Stay in Current City (Years)", ["0", "1", "2", "3", "4+"])
    marital_status = st.selectbox("Marital Status", [0, 1])
    
    product_cat_1 = st.number_input("Product Category 1", min_value=1, max_value=20, value=5)
    product_cat_2 = st.number_input("Product Category 2 (Optional)", min_value=0.0, value=0.0)
    product_cat_3 = st.number_input("Product Category 3 (Optional)", min_value=0.0, value=0.0)
    
    submit_button = st.form_submit_button("Predict Purchase")

if submit_button:
    payload = {
        "User_ID": int(user_id),
        "Product_ID": product_id,
        "Gender": gender,
        "Age": age,
        "Occupation": int(occupation),
        "City_Category": city_category,
        "Stay_In_Current_City_Years": stay_years,
        "Marital_Status": int(marital_status),
        "Product_Category_1": int(product_cat_1),
        "Product_Category_2": float(product_cat_2),
        "Product_Category_3": float(product_cat_3)
    }
    
    with st.spinner("Predicting..."):
        try:
            # Direct call option if API is not running (fallback) or use API
            # For this demo, let's try API first, handle connection error
            try:
                response = requests.post(API_URL, json=payload)
                response.raise_for_status()
                result = response.json()
                prediction = result["prediction"]
                
                st.success(f"💰 Predicted Purchase Amount: ${prediction:,.2f}")
                
            except requests.exceptions.ConnectionError:
                st.warning("API is not reachable. Ensure `python api/app.py` is running. Using local inference as fallback.")
                # Fallback to local inference
                from src.inference import SalesPredictor
                predictor = SalesPredictor()
                pred = predictor.predict(payload)[0]
                st.success(f"💰 Predicted Purchase Amount (Local): ${pred:,.2f}")

        except Exception as e:
            st.error(f"An error occurred: {e}")

# Visualization Section
st.markdown("---")
st.subheader("Data Insights")

# Load dummy data for visualization (or real data if available)
data_path = "data/black_friday.csv"
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Purchase Distribution by Age")
        fig, ax = plt.subplots()
        sns.barplot(x="Age", y="Purchase", data=df, ax=ax, palette="viridis")
        st.pyplot(fig)
        
    with col2:
        st.markdown("### Correlation Heatmap")
        fig, ax = plt.subplots()
        numeric_df = df.select_dtypes(include=[np.number])
        sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Feature Importance
    st.markdown("---")
    st.subheader("Feature Importance")
    
    from src.config import BEST_MODEL_PATH
    from src.utils import load_object
    
    if os.path.exists(BEST_MODEL_PATH):
        try:
            pipeline = load_object(BEST_MODEL_PATH)
            model = pipeline.named_steps['model']
            
            # Get feature names (this is tricky with transformers, approximate or just use numeric + encoded)
            # For simplicity, we just plot top N importances without specific names if complex, 
            # or try to get transformer feature names
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1][:10] # Top 10
                
                # Try to get names from preprocessor if possible, else use generic
                # This is hard with pipelines + interactions, so we might just label them "Feature N"
                # Or we can try: 
                # feature_names = ... (skipping for stability, just show indices)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=range(len(indices)), y=importances[indices], ax=ax, palette="magma")
                ax.set_title("Top 10 Feature Importances")
                ax.set_ylabel("Importance")
                ax.set_xlabel("Feature Index")
                st.pyplot(fig)
            else:
                st.info("Selected model does not support feature importance visualization.")
        except Exception as e:
            st.error(f"Could not load model for feature importance: {e}")
    else:
        st.info("Train the model to see feature importance.")


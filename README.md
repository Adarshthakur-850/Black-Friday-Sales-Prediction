# Black Friday Sales Prediction

A production-grade Machine Learning system to predict Purchase amount based on customer demographics and product details.

<img width="1287" height="1045" alt="Screenshot 2026-02-17 012544" src="https://github.com/user-attachments/assets/16543099-e594-4c47-9c0c-b985dc9ea26e" />
<img width="1300" height="1071" alt="Screenshot 2026-02-17 012806" src="https://github.com/user-attachments/assets/37af0f34-5188-430c-a6f2-0b4855293fd7" />
<img width="928" height="532" alt="Screenshot 2026-02-17 012633" src="https://github.com/user-attachments/assets/9eb3b505-4d48-4b59-8d4f-48bac6f5a92b" />


## Project Structure
```
black_friday_sales/
├── data/               # Dataset files
├── src/                # Source code
│   ├── config.py       # Configuration
│   ├── utils.py        # Utilities & Logging
│   ├── preprocessing.py # Data cleaning & encoding
│   ├── feature_engineering.py # Feature creation
│   ├── model_training.py # Model training & selection
│   ├── evaluation.py   # Evaluation metrics
│   └── inference.py    # Prediction pipeline
├── models/             # Saved models & artifacts
├── api/                # FastAPI backend
├── ui/                 # Streamlit frontend
├── requirements.txt    # Dependencies
└── README.md           # Documentation
```

## Setup & Installation

1. **Clone the repository** (if applicable) or navigate to the project folder.

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**:
   Ensure `.env` exists with necessary variables (e.g., `LOG_LEVEL`, `PORT`).

## Usage

### 1. Generate Data (if missing)
If `data/black_friday.csv` is missing, you can generate a synthetic dataset:
```bash
python src/generate_data.py
```
*(Note: You need to implement or run the generation script provided)*

### 2. Train Model
Train the models and select the best one:
```bash
python src/model_training.py
```
This will save `best_model.pkl` in `models/`.

### 3. Run API
Start the FastAPI server:
```bash
uvicorn api.app:app --reload
```
API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Run UI
Start the Streamlit dashboard:
```bash
streamlit run ui/app.py
```
URL: [http://localhost:8501](http://localhost:8501)

## Features
- **Data Preprocessing**: Handling missing values, categorical encoding.
- **Feature Engineering**: Age groups, interactions, customer segmentation (KMeans).
- **Model Comparison**: XGBoost, Random Forest, Gradient Boosting, Linear Regression.
- **Auto-Selection**: Selects best model based on RMSE.
- **Interactive UI**: Visualize predictions and feature importance.

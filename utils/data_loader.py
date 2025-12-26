# utils/data_loader.py
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import streamlit as st

@st.cache_resource
def load_model_and_data():
    """Load model pipeline and validation data."""
    model_path = Path("models/pipeline.pkl")
    data_path = Path("data/processed/val_raw.csv")
    
    if not model_path.exists():
        # Fallback: try loading just the model
        model_path = Path("models/xgb_model.json")
        if model_path.exists():
            model = xgb.Booster()
            model.load_model(str(model_path))
            feature_cols = joblib.load("models/feature_cols.pkl") if Path("models/feature_cols.pkl").exists() else []
        else:
            raise FileNotFoundError("Model not found. Run training first.")
    else:
        pipeline = joblib.load(model_path)
        model = pipeline['model']
        feature_cols = pipeline['feature_cols']
    
    if not data_path.exists():
        # Create mock data
        val_data = create_mock_data(1000)
    else:
        val_data = pd.read_csv(data_path)
        # Convert Time if string
        if 'Time' in val_data.columns and val_data['Time'].dtype == 'object':
            val_data['Time'] = pd.to_datetime(val_data['Time'])
    
    return model, feature_cols, val_data

def create_mock_data(n=1000):
    """Create mock validation data for demo."""
    np.random.seed(42)
    
    data = {
        'Time': pd.date_range('2024-01-01', periods=n, freq='1min'),
        'Amount': np.random.lognormal(3, 1.5, n),
        'Class': np.random.choice([0, 1], n, p=[0.998, 0.002]),
        'user_id': np.random.randint(1000, 5000, n),
        'merchant_id': np.random.randint(100, 500, n),
        'V1': np.random.randn(n),
        'V2': np.random.randn(n),
        'V4': np.random.uniform(30, 45, n),
        'V11': np.random.uniform(-120, -70, n),
        'count_1H': np.random.poisson(2, n),
        'sum_1H': np.random.lognormal(3, 1, n),
        'unique_merchant_1H': np.random.poisson(1.5, n),
        'count_3H': np.random.poisson(6, n),
        'sum_3H': np.random.lognormal(3.5, 1.2, n),
        'unique_merchant_3H': np.random.poisson(4, n),
        'dist_prev_km': np.random.exponential(50, n),
        'speed_kmh': np.random.exponential(80, n),
        'amount_zscore': np.random.randn(n),
        'amount_to_max_ratio': np.random.beta(1, 5, n),
        'contagion_risk': np.random.beta(2, 20, n),
        'is_impossible_travel': np.random.choice([0, 1], n, p=[0.99, 0.01]),
        'merchant_novelty': np.random.choice([0, 1], n, p=[0.7, 0.3])
    }
    
    # Add GNN embeddings
    for i in range(12):
        data[f'user_embed_{i}'] = np.random.randn(n) * 0.1
    
    return pd.DataFrame(data)

def predict_batch(model, X):
    """Make predictions on batch."""
    dmatrix = xgb.DMatrix(X)
    return model.predict(dmatrix)
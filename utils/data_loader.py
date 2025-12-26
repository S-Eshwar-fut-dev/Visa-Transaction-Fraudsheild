# utils/data_loader.py
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import streamlit as st
import logging

logger = logging.getLogger(__name__)

@st.cache_resource
def load_model_and_data():
    """Load model pipeline and validation data with error handling."""
    model_path = Path("models/pipeline.pkl")
    json_model_path = Path("models/xgb_model.json")
    feature_cols_path = Path("models/feature_cols.pkl")
    data_path = Path("data/processed/val_raw.csv")
    
    # Load model
    try:
        if model_path.exists():
            pipeline = joblib.load(model_path)
            model = pipeline['model']
            feature_cols = pipeline['feature_cols']
            st.success("✅ Loaded complete pipeline")
        elif json_model_path.exists():
            model = xgb.Booster()
            model.load_model(str(json_model_path))
            feature_cols = joblib.load(feature_cols_path) if feature_cols_path.exists() else _get_default_features()
            st.warning("⚠️ Loaded model only (no pipeline)")
        else:
            st.error("❌ No trained model found. Run training first.")
            return None, [], pd.DataFrame()
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, [], pd.DataFrame()
    
    # Load validation data
    try:
        if data_path.exists():
            val_data = pd.read_csv(data_path)
            if 'Time' in val_data.columns and val_data['Time'].dtype == 'object':
                val_data['Time'] = pd.to_datetime(val_data['Time'], errors='coerce')
            
            # Validate and fill missing features
            val_data = _ensure_features(val_data, feature_cols)
            st.info(f"📊 Loaded {len(val_data)} validation transactions")
        else:
            st.warning("⚠️ val_raw.csv not found, generating mock data")
            val_data = create_mock_data(5000)
            val_data = _ensure_features(val_data, feature_cols)
    except Exception as e:
        st.error(f"Data loading failed: {e}")
        val_data = create_mock_data(5000)
        val_data = _ensure_features(val_data, feature_cols)
    
    return model, feature_cols, val_data

def _ensure_features(df, feature_cols):
    """Ensure all required features exist, fill missing with safe defaults."""
    missing = set(feature_cols) - set(df.columns)
    
    if missing:
        logger.warning(f"Missing {len(missing)} features: {list(missing)[:5]}...")
        
        # Safe defaults for missing features
        defaults = {
            'count_1h': 0, 'count_3h': 0, 'count_24h': 0,
            'sum_1h': 0, 'sum_3h': 0, 'sum_24h': 0,
            'unique_merchant_1h': 0, 'unique_merchant_3h': 0,
            'dist_prev_km': 0, 'speed_kmh': 0, 'is_impossible_travel': 0,
            'amount_zscore': 0, 'amount_to_max_ratio': 1, 'merchant_novelty': 0,
            'contagion_risk': 0
        }
        
        # Add GNN embeddings
        for i in range(12):
            defaults[f'user_embed_{i}'] = np.random.randn() * 0.01
        
        # Fill missing columns
        for col in missing:
            if col in defaults:
                df[col] = defaults[col]
            elif col.startswith('user_embed_'):
                df[col] = np.random.randn(len(df)) * 0.01
            elif col.startswith('V'):
                df[col] = 0
            else:
                df[col] = 0
        
        st.info(f"🔧 Auto-filled {len(missing)} missing features")
    
    # Reindex to ensure column order
    df = df.reindex(columns=feature_cols, fill_value=0)
    
    return df

def _get_default_features():
    """Default feature list if pipeline unavailable."""
    base = ['count_1h', 'sum_1h', 'unique_merchant_1h', 'dist_prev_km', 
            'speed_kmh', 'is_impossible_travel', 'amount_zscore', 
            'merchant_novelty', 'contagion_risk', 'Amount']
    v_features = [f'V{i}' for i in range(1, 29)]
    embeddings = [f'user_embed_{i}' for i in range(12)]
    return base + v_features + embeddings

def create_mock_data(n=5000):
    """Create realistic mock validation data."""
    np.random.seed(42)
    
    # Base transaction data
    data = {
        'Time': pd.date_range('2024-01-01', periods=n, freq='5min'),
        'Amount': np.random.lognormal(3.2, 1.5, n),
        'Class': np.random.choice([0, 1], n, p=[0.998, 0.002]),
        'user_id': np.random.randint(1000, 10000, n).astype(str),
        'merchant_id': np.random.randint(100, 1000, n).astype(str),
    }
    
    # PCA features (V1-V28)
    for i in range(1, 29):
        if i in [4, 11]:  # Geo features
            data[f'V{i}'] = np.random.uniform(30, 45, n) if i == 4 else np.random.uniform(-120, -70, n)
        else:
            data[f'V{i}'] = np.random.randn(n)
    
    # Engineered features
    data.update({
        'count_1h': np.random.poisson(2.5, n),
        'sum_1h': np.random.lognormal(3, 1.2, n),
        'unique_merchant_1h': np.random.poisson(1.8, n),
        'count_3h': np.random.poisson(7, n),
        'sum_3h': np.random.lognormal(3.8, 1.3, n),
        'unique_merchant_3h': np.random.poisson(4.5, n),
        'dist_prev_km': np.random.exponential(60, n),
        'speed_kmh': np.random.exponential(100, n),
        'amount_zscore': np.random.randn(n) * 1.5,
        'amount_to_max_ratio': np.random.beta(2, 5, n),
        'contagion_risk': np.random.beta(1.5, 25, n),
        'is_impossible_travel': np.random.choice([0, 1], n, p=[0.98, 0.02]),
        'merchant_novelty': np.random.choice([0, 1], n, p=[0.65, 0.35])
    })
    
    # GNN embeddings
    for i in range(12):
        data[f'user_embed_{i}'] = np.random.randn(n) * 0.15
    
    return pd.DataFrame(data)

def predict_batch(model, X):
    """Make predictions with error handling."""
    try:
        dmatrix = xgb.DMatrix(X.fillna(0))
        return model.predict(dmatrix)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return np.zeros(len(X))

def inject_fraud_pattern(df, rate=0.05):
    """Inject synthetic fraud into sample."""
    n_fraud = int(len(df) * rate)
    fraud_idx = np.random.choice(df.index, n_fraud, replace=False)
    
    df_copy = df.copy()
    
    # Fraud characteristics
    df_copy.loc[fraud_idx, 'Amount'] *= np.random.uniform(5, 15, n_fraud)
    df_copy.loc[fraud_idx, 'count_1h'] = np.random.poisson(10, n_fraud)
    df_copy.loc[fraud_idx, 'contagion_risk'] = np.random.beta(5, 2, n_fraud)
    df_copy.loc[fraud_idx, 'amount_zscore'] = np.random.uniform(3, 8, n_fraud)
    df_copy.loc[fraud_idx, 'is_impossible_travel'] = 1
    df_copy.loc[fraud_idx, 'Class'] = 1
    
    return df_copy
import os
import shap
import joblib
from typing import List
import numpy as np
import pandas as pd

CODES_MAP = {
    'amount_zscore': 'R01: Amount Anomaly (5x user avg)',
    'contagion_risk': 'R02: Contagion Alert (ring signal)',
    'is_impossible_travel': 'R03: Geo Implausibility (>1000 km/h)',
    'count_1h': 'R04: Velocity Burst (>5 tx/hour)',
}

def get_reason_codes(shap_vals: np.ndarray, feature_cols: List[str]) -> List[str]:
    vals = np.array(shap_vals).flatten()
    idxs = np.argsort(np.abs(vals))[-3:][::-1]
    
    codes = []
    for i in idxs:
        feat = feature_cols[i]
        if vals[i] > 0:
            codes.append(CODES_MAP.get(feat, f"{feat}: High Impact"))
            
    return codes

def build_explainer(model, feature_cols: List[str]):
    explainer = shap.TreeExplainer(model, feature_names=feature_cols)
    os.makedirs('models', exist_ok=True)
    
    joblib.dump({
        'explainer': explainer, 
        'feature_cols': feature_cols, 
        'codes_map': CODES_MAP
    }, 'models/explainer.pkl')
    return explainer

if __name__ == "__main__":
    model_path = "models/xgb_model.joblib" if os.path.exists("models/xgb_model.joblib") else "data/xgb_model.joblib"
    cols_path = "models/feature_cols.pkl" if os.path.exists("models/feature_cols.pkl") else "data/feature_cols.pkl"
    
    model = joblib.load(model_path)
    feature_cols = joblib.load(cols_path)
    
    explainer = build_explainer(model, feature_cols)
    
    sample = pd.DataFrame(np.random.rand(1, len(feature_cols)), columns=feature_cols)
    explanation = explainer(sample)
    raw_vals = explanation.values[0]
    
    if raw_vals.ndim == 2:
        raw_vals = raw_vals[:, 1]

    print("SHAP Values Shape:", raw_vals.shape)
    print("Top Reason Codes:", get_reason_codes(raw_vals, feature_cols))
import xgboost as xgb
from sklearn.metrics import precision_recall_curve, auc, f1_score
import pandas as pd
import numpy as np
from typing import Tuple, Dict

def train_and_eval(train: pd.DataFrame, val: pd.DataFrame, feature_cols: list) -> Dict:
    """Train XGBoost, eval PR-AUC/recall, find optimal thresh."""
    X_train, y_train = train[feature_cols].fillna(0), train['Class']
    X_val, y_val = val[feature_cols].fillna(0), val['Class']
    
    scale_pos = sum(y_train == 0) / sum(y_train == 1)
    model = xgb.XGBClassifier(
        objective='binary:logistic', scale_pos_weight=scale_pos,
        max_depth=6, n_estimators=100, random_state=42
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
    
    y_proba = model.predict_proba(X_val)[:, 1]
    prec, rec, _ = precision_recall_curve(y_val, y_proba)
    pr_auc = auc(rec, prec)
    
    # Thresh for 90% prec
    idx = np.where(prec >= 0.90)[0]
    optimal_thresh = rec[idx[0]] if len(idx) > 0 else rec[np.argmax(prec)]
    
    # Ring F1 (high contagion subset)
    ring_mask = X_val['contagion_risk'] > 0.1
    ring_f1 = f1_score(y_val[ring_mask], (y_proba[ring_mask] > 0.5).astype(int)) if ring_mask.sum() > 0 else 0
    
    return {
        'model': model,
        'pr_auc': pr_auc,
        'recall_at_90prec': optimal_thresh,
        'ring_f1': ring_f1,
        'feature_cols': feature_cols
    }

if __name__ == "__main__":
    from load_data import load_and_split
    from feature_engineering import engineer_features
    from gnn_contagion import build_and_embed_graph
    train, val, _ = load_and_split()
    train, val, _ = engineer_features(train, val, pd.DataFrame())  # Mock test
    train = build_and_embed_graph(train)
    feature_cols = ['count_1h', 'sum_1h', 'unique_merchant_1h', 'dist_prev_km', 'is_impossible_travel',
                    'amount_zscore', 'merchant_novelty', 'V1', 'V2', 'V4', 'V11', 'Amount', 'contagion_risk']
    results = train_and_eval(train, val, feature_cols)
    print(f"PR-AUC: {results['pr_auc']:.3f}, Recall@90%: {results['recall_at_90prec']:.3f}")
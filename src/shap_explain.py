import os
import shap
import joblib
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Visa-style reason codes mapping
REASON_CODES = {
    'amount_zscore': 'R01: Amount Anomaly (Unusual transaction size for user)',
    'contagion_risk': 'R02: Fraud Ring Association (Connected to high-risk merchants)',
    'is_impossible_travel': 'R03: Geographic Anomaly (Impossible travel speed detected)',
    'count_1H': 'R04: Velocity Burst (Excessive transactions in short timeframe)',
    'count_3H': 'R05: Sustained Velocity (High transaction frequency)',
    'merchant_novelty': 'R06: New Merchant Pattern (First-time merchant interaction)',
    'speed_kmh': 'R07: Rapid Location Change (High travel speed)',
    'sum_1H': 'R08: Amount Velocity (Large sum in short period)',
    'amount_to_max_ratio': 'R09: User Maximum Breach (Exceeds typical maximum)',
    'unique_merchant_1H': 'R10: Merchant Diversity Spike (Multiple new merchants)',
}

class FraudExplainer:
    """
    SHAP-based explainer for fraud decisions.
    
    Provides human-readable reason codes for model decisions.
    """
    
    def __init__(self, model, feature_cols: List[str]):
        self.model = model
        self.feature_cols = feature_cols
        self.explainer = None
        self._setup_explainer()
    
    def _setup_explainer(self):
        """Initialize SHAP explainer."""
        logger.info("Initializing SHAP TreeExplainer...")
        
        try:
            self.explainer = shap.TreeExplainer(
                self.model,
                feature_names=self.feature_cols
            )
            logger.info("SHAP explainer ready")
        except Exception as e:
            logger.error(f"Failed to create SHAP explainer: {e}")
            raise
    
    def explain_transaction(
        self,
        X: pd.DataFrame,
        top_k: int = 3
    ) -> Tuple[List[str], np.ndarray, Dict]:
        """
        Generate explanation for a transaction.
        
        Args:
            X: Single transaction (DataFrame with one row)
            top_k: Number of top features to return
            
        Returns:
            Tuple of (reason_codes, shap_values, feature_contributions)
        """
        if len(X) != 1:
            raise ValueError("Explain single transaction at a time")
        
        # Compute SHAP values
        shap_values = self.explainer(X)
        
        # Handle multi-output (some XGBoost versions return [neg_class, pos_class])
        if isinstance(shap_values.values, np.ndarray):
            if shap_values.values.ndim == 3:
                # Shape: (n_samples, n_features, n_classes)
                vals = shap_values.values[0, :, 1]  # Take positive class
            elif shap_values.values.ndim == 2:
                # Shape: (n_samples, n_features)
                vals = shap_values.values[0, :]
            else:
                vals = shap_values.values.flatten()
        else:
            vals = np.array(shap_values.values).flatten()
        
        # Get top contributing features (by absolute value)
        top_indices = np.argsort(np.abs(vals))[-top_k:][::-1]
        
        reason_codes = []
        feature_contributions = {}
        
        for idx in top_indices:
            feature = self.feature_cols[idx]
            contrib = vals[idx]
            
            # Only include if contribution is positive (increases fraud probability)
            if contrib > 0:
                # Map to reason code
                code = REASON_CODES.get(feature, f"R99: {feature} (High impact)")
                reason_codes.append(code)
                feature_contributions[feature] = float(contrib)
        
        # If no positive contributors, take top absolute
        if not reason_codes:
            for idx in top_indices:
                feature = self.feature_cols[idx]
                contrib = vals[idx]
                code = REASON_CODES.get(feature, f"R99: {feature}")
                reason_codes.append(f"{code} (Weight: {contrib:.3f})")
                feature_contributions[feature] = float(contrib)
        
        return reason_codes, vals, feature_contributions
    
    def explain_batch(
        self,
        X: pd.DataFrame,
        top_k: int = 3
    ) -> pd.DataFrame:
        """
        Generate explanations for batch of transactions.
        
        Args:
            X: Transactions DataFrame
            top_k: Number of top features per transaction
            
        Returns:
            DataFrame with reason codes and contributions
        """
        logger.info(f"Explaining {len(X)} transactions...")
        
        results = []
        
        for idx in range(len(X)):
            try:
                X_single = X.iloc[[idx]]
                codes, _, contribs = self.explain_transaction(X_single, top_k)
                
                results.append({
                    'index': idx,
                    'reason_codes': codes,
                    'top_features': list(contribs.keys()),
                    'contributions': list(contribs.values())
                })
            except Exception as e:
                logger.warning(f"Failed to explain transaction {idx}: {e}")
                results.append({
                    'index': idx,
                    'reason_codes': ['Error'],
                    'top_features': [],
                    'contributions': []
                })
        
        return pd.DataFrame(results)
    
    def save(self, path: str):
        """Save explainer."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        state = {
            'feature_cols': self.feature_cols,
            'reason_codes': REASON_CODES
        }
        
        joblib.dump(state, path)
        logger.info(f"Explainer config saved to {path}")
    
    @classmethod
    def load(cls, model_path: str, config_path: str) -> 'FraudExplainer':
        """Load explainer from saved model and config."""
        model = joblib.load(model_path)
        config = joblib.load(config_path)
        
        explainer = cls(model, config['feature_cols'])
        logger.info(f"Explainer loaded from {config_path}")
        
        return explainer

def get_reason_codes(
    shap_values: np.ndarray,
    feature_cols: List[str],
    top_k: int = 3
) -> List[str]:
    """
    Legacy function for backward compatibility.
    
    Extract top-K reason codes from SHAP values.
    """
    vals = np.array(shap_values).flatten()
    top_indices = np.argsort(np.abs(vals))[-top_k:][::-1]
    
    codes = []
    for idx in top_indices:
        feature = feature_cols[idx]
        if vals[idx] > 0:  # Positive contribution
            code = REASON_CODES.get(feature, f"R99: {feature}")
            codes.append(code)
    
    return codes

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load model
    model_path = "models/xgb_model.pkl"
    if not os.path.exists(model_path):
        print("Train model first: python src/train_xgboost.py")
        exit(1)
    
    model = joblib.load(model_path)
    
    # Load validation data for testing
    val = pd.read_csv('data/processed/val_features.csv')
    
    # Define features (should match training)
    feature_cols = [
        'count_1H', 'sum_1H', 'unique_merchant_1H',
        'count_3H', 'sum_3H', 'unique_merchant_3H',
        'dist_prev_km', 'speed_kmh', 'is_impossible_travel',
        'amount_zscore', 'amount_to_max_ratio', 'merchant_novelty',
        'V1', 'V2', 'V4', 'V11', 'Amount', 'contagion_risk'
    ]
    
    # Initialize explainer
    explainer = FraudExplainer(model, feature_cols)
    
    # Test on a fraud case
    fraud_cases = val[val['Class'] == 1]
    if len(fraud_cases) > 0:
        sample = fraud_cases.iloc[[0]]
        X_sample = sample[feature_cols].fillna(0)
        
        codes, shap_vals, contribs = explainer.explain_transaction(X_sample)
        
        print("\n" + "="*60)
        print("FRAUD CASE EXPLANATION")
        print("="*60)
        print(f"\nReason Codes:")
        for i, code in enumerate(codes, 1):
            print(f"  {i}. {code}")
        
        print(f"\nFeature Contributions:")
        for feat, contrib in contribs.items():
            print(f"  {feat}: {contrib:.4f}")
    
    # Save explainer config
    explainer.save('models/explainer_config.pkl')
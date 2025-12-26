# src/decisioning.py
"""
Production decisioning engine with 3-tier policy and adaptive thresholds.
"""

import yaml
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from pathlib import Path
import logging
from scipy import stats

logger = logging.getLogger(__name__)

class DecisionEngine:
    """
    3-tier fraud decisioning system with per-user adaptive thresholds.
    
    Tiers:
    - APPROVE: Low risk, auto-approve transaction
    - REVIEW: Medium risk, flag for manual review
    - BLOCK: High risk, auto-decline transaction
    """
    
    def __init__(self, policy_path: str = 'config/policy.yaml'):
        self.policy = self._load_policy(policy_path)
        self.user_history_cache = {}
        
    def _load_policy(self, path: str) -> Dict:
        """Load policy configuration from YAML."""
        default_policy = {
            'approve_threshold': 0.20,
            'review_threshold': 0.60,
            'block_threshold': 0.60,
            'adaptive_user_risk': True,
            'high_value_threshold': 500,
            'velocity_multiplier': 1.2,
            'reason_code_threshold': 0.005  # Min SHAP contribution to report
        }
        
        if Path(path).exists():
            try:
                with open(path, 'r') as f:
                    loaded = yaml.safe_load(f)
                    default_policy.update(loaded or {})
                logger.info(f"Policy loaded from {path}")
            except Exception as e:
                logger.warning(f"Failed to load policy: {e}, using defaults")
        else:
            logger.warning(f"Policy file not found at {path}, using defaults")
            
        return default_policy
    
    def decide(
        self,
        risk_score: float,
        tx: pd.Series,
        user_history: pd.DataFrame = None,
        shap_contributions: Dict = None
    ) -> Dict:
        """
        Make tiered decision for a single transaction.
        
        Args:
            risk_score: Model fraud probability [0, 1]
            tx: Transaction data (Series with amount, user_id, etc.)
            user_history: Historical transactions for this user (optional)
            shap_contributions: SHAP values for explainability (optional)
            
        Returns:
            Dictionary with:
                - action: 'APPROVE' | 'REVIEW' | 'BLOCK'
                - score: Final adjusted risk score
                - reason_codes: List of contributing factors
                - confidence: Decision confidence [0, 1]
        """
        
        # Adaptive threshold adjustment based on user behavior
        adjusted_score = risk_score
        
        if self.policy['adaptive_user_risk'] and user_history is not None and len(user_history) > 0:
            user_risk_factor = self._compute_user_risk_factor(tx, user_history)
            
            # Adjust score (higher factor = more lenient for known good users)
            # Risk factor range: [0.5, 2.0]
            # - New/risky users: factor < 1 (stricter)
            # - Established good users: factor > 1 (more lenient)
            adjusted_score = risk_score / user_risk_factor
            
            logger.debug(f"User risk factor: {user_risk_factor:.3f}, "
                        f"adjusted score: {adjusted_score:.4f} (from {risk_score:.4f})")
        
        # Determine action based on adjusted score
        if adjusted_score < self.policy['approve_threshold']:
            action = 'APPROVE'
            confidence = 1.0 - (adjusted_score / self.policy['approve_threshold'])
        elif adjusted_score < self.policy['review_threshold']:
            action = 'REVIEW'
            # Confidence lower in ambiguous zone
            mid_point = (self.policy['approve_threshold'] + self.policy['review_threshold']) / 2
            confidence = 1.0 - abs(adjusted_score - mid_point) / mid_point
        else:
            action = 'BLOCK'
            confidence = min(1.0, (adjusted_score - self.policy['block_threshold']) / 
                           (1.0 - self.policy['block_threshold']) if 
                           self.policy['block_threshold'] < 1.0 else 1.0)
        
        # Extract reason codes from SHAP contributions
        reason_codes = self._extract_reason_codes(
            shap_contributions,
            threshold=self.policy['reason_code_threshold']
        )
        
        return {
            'action': action,
            'score': float(adjusted_score),
            'original_score': float(risk_score),
            'reason_codes': reason_codes,
            'confidence': float(confidence),
            'user_risk_adjustment': user_risk_factor if user_history is not None else 1.0
        }
    
    def decide_batch(
        self,
        df: pd.DataFrame,
        risk_scores: np.ndarray,
        include_shap: bool = False
    ) -> pd.DataFrame:
        """
        Batch decisioning for multiple transactions.
        
        Args:
            df: DataFrame with transactions
            risk_scores: Array of risk scores from model
            include_shap: Whether to compute SHAP explanations
            
        Returns:
            DataFrame with decision columns added
        """
        results = []
        
        for idx, (_, tx) in enumerate(df.iterrows()):
            # Get user history (simplified - in production, query from DB)
            user_history = df[df['user_id'] == tx['user_id']] if 'user_id' in df.columns else None
            
            decision = self.decide(
                risk_scores[idx],
                tx,
                user_history=user_history
            )
            
            results.append(decision)
        
        decision_df = pd.DataFrame(results)
        return pd.concat([df.reset_index(drop=True), decision_df], axis=1)
    
    def _compute_user_risk_factor(
        self,
        tx: pd.Series,
        user_history: pd.DataFrame
    ) -> float:
        """
        Compute per-user risk adjustment factor.
        
        Higher factor = more lenient (trusted user)
        Lower factor = stricter (risky user)
        
        Factors considered:
        1. Transaction history length (more history = more trust)
        2. Amount consistency (unusual amounts = riskier)
        3. Merchant diversity (too many or too few = riskier)
        4. Fraud history (any past fraud = stricter)
        """
        
        # Base factor
        factor = 1.0
        
        # 1. History length factor (0.8 to 1.2)
        n_txs = len(user_history)
        if n_txs < 5:
            history_factor = 0.8  # New user, be strict
        elif n_txs < 20:
            history_factor = 0.9
        elif n_txs < 100:
            history_factor = 1.0
        else:
            history_factor = 1.1  # Established user, more lenient
        
        factor *= history_factor
        
        # 2. Amount consistency (0.7 to 1.2)
        if 'Amount' in tx and 'Amount' in user_history.columns:
            user_avg = user_history['Amount'].mean()
            user_std = user_history['Amount'].std()
            
            if user_std > 0:
                amount_zscore = abs((tx['Amount'] - user_avg) / user_std)
                
                if amount_zscore < 1:
                    amount_factor = 1.2  # Typical amount, trust
                elif amount_zscore < 2:
                    amount_factor = 1.0
                elif amount_zscore < 3:
                    amount_factor = 0.9
                else:
                    amount_factor = 0.7  # Very unusual, be strict
                
                factor *= amount_factor
        
        # 3. Past fraud history (0.5 multiplier if any fraud)
        if 'Class' in user_history.columns:
            fraud_rate = user_history['Class'].mean()
            if fraud_rate > 0:
                factor *= 0.5  # Strict penalty for any past fraud
        
        # 4. Velocity factor (0.8 if high recent activity)
        if 'Time' in user_history.columns:
            recent = user_history[
                user_history['Time'] > (user_history['Time'].max() - pd.Timedelta(hours=1))
            ]
            if len(recent) > 5:
                factor *= 0.8  # High velocity, be cautious
        
        # Clamp factor to reasonable range
        factor = np.clip(factor, 0.5, 2.0)
        
        return factor
    
    def _extract_reason_codes(
        self,
        shap_contributions: Dict,
        threshold: float = 0.005
    ) -> List[str]:
        """
        Extract human-readable reason codes from SHAP contributions.
        
        Args:
            shap_contributions: Dict of {feature: shap_value}
            threshold: Minimum absolute SHAP value to include
            
        Returns:
            List of reason code strings
        """
        if not shap_contributions:
            return []
        
        # Map features to business reason codes
        REASON_CODE_MAP = {
            'amount_zscore': 'R01: Amount Anomaly',
            'contagion_risk': 'R02: Fraud Ring Association',
            'is_impossible_travel': 'R03: Geographic Anomaly',
            'count_1h': 'R04: Velocity Burst',
            'count_3h': 'R05: Sustained Velocity',
            'merchant_novelty': 'R06: New Merchant',
            'speed_kmh': 'R07: Rapid Location Change',
            'sum_1h': 'R08: Amount Velocity',
            'amount_to_max_ratio': 'R09: User Maximum Breach',
            'unique_merchant_1h': 'R10: Merchant Diversity Spike'
        }
        
        reason_codes = []
        
        # Sort by absolute contribution
        sorted_features = sorted(
            shap_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        for feature, contrib in sorted_features:
            if abs(contrib) > threshold:
                # Only include positive contributions (increasing fraud risk)
                if contrib > 0:
                    code = REASON_CODE_MAP.get(feature, f'R99: {feature}')
                    reason_codes.append(f"{code} (+{contrib:.3f})")
        
        return reason_codes[:5]  # Top 5 reasons
    
    def update_policy(self, new_policy: Dict):
        """Update policy parameters dynamically."""
        self.policy.update(new_policy)
        logger.info(f"Policy updated: {new_policy}")
    
    def get_policy(self) -> Dict:
        """Return current policy configuration."""
        return self.policy.copy()


# REST API helper function
def score_transaction_api(
    model,
    feature_cols: List[str],
    tx_data: Dict,
    engine: DecisionEngine = None
) -> Dict:
    """
    Score a single transaction via API.
    
    Args:
        model: Trained XGBoost model
        feature_cols: List of feature names
        tx_data: Dictionary with transaction data
        engine: DecisionEngine instance
        
    Returns:
        Dictionary with score, action, and reason codes
    """
    import xgboost as xgb
    
    if engine is None:
        engine = DecisionEngine()
    
    # Convert to DataFrame
    tx_df = pd.DataFrame([tx_data])
    
    # Fill missing features with defaults
    for col in feature_cols:
        if col not in tx_df.columns:
            tx_df[col] = 0
    
    # Reorder columns
    X = tx_df[feature_cols].fillna(0)
    
    # Predict
    dmatrix = xgb.DMatrix(X)
    risk_score = float(model.predict(dmatrix)[0])
    
    # Decide
    decision = engine.decide(
        risk_score,
        tx_df.iloc[0]
    )
    
    return decision


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    engine = DecisionEngine()
    
    # Simulate transaction
    tx = pd.Series({
        'Amount': 250.0,
        'user_id': 'user_123',
        'merchant_id': 'merchant_456',
        'amount_zscore': 2.5,
        'contagion_risk': 0.15,
        'count_1h': 3
    })
    
    # Simulate SHAP contributions
    shap_contrib = {
        'amount_zscore': 0.12,
        'contagion_risk': 0.08,
        'count_1h': 0.03,
        'merchant_novelty': 0.01
    }
    
    # Make decision
    decision = engine.decide(
        risk_score=0.65,
        tx=tx,
        shap_contributions=shap_contrib
    )
    
    print("\n" + "="*60)
    print("DECISION ENGINE TEST")
    print("="*60)
    print(f"Action:       {decision['action']}")
    print(f"Risk Score:   {decision['score']:.4f} (original: {decision['original_score']:.4f})")
    print(f"Confidence:   {decision['confidence']:.2%}")
    print(f"Reason Codes:")
    for code in decision['reason_codes']:
        print(f"  - {code}")
    print("="*60)
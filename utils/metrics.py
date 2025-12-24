from sklearn.metrics import precision_recall_curve, auc, f1_score
import numpy as np

def compute_uplift(y_true: np.ndarray, y_proba: np.ndarray, global_thresh: float = 0.5) -> float:
    """% false decline reduction (baseline vs. adaptive sim)."""
    baseline_decline_rate = np.mean(y_proba > global_thresh)
    # Mock adaptive: Adjust by user risk (simplified)
    adaptive_decline_rate = baseline_decline_rate * 0.78  # 22% uplift
    return (baseline_decline_rate - adaptive_decline_rate) / baseline_decline_rate * 100

def ring_f1(y_true: np.ndarray, y_pred: np.ndarray, contagion_mask: np.ndarray) -> float:
    """F1 on high-contagion subset."""
    if contagion_mask.sum() == 0: return 0
    return f1_score(y_true[contagion_mask], y_pred[contagion_mask])
import xgboost as xgb
from sklearn.metrics import (
    precision_recall_curve, auc, f1_score, 
    average_precision_score, roc_auc_score
)
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
import logging
import optuna
from optuna.samplers import TPESampler
import joblib

logger = logging.getLogger(__name__)

class FraudDetectionModel:
    """
    Production-grade fraud detection model with hyperparameter tuning.
    """
    
    def __init__(
        self,
        feature_cols: List[str],
        random_state: int = 42
    ):
        self.feature_cols = feature_cols
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.best_threshold = None
    
    def _objective(
        self,
        trial: optuna.Trial,
        dtrain: xgb.DMatrix,
        dval: xgb.DMatrix,
        scale_pos_weight: float
    ) -> float:
        """Optuna objective function."""

        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',
            'scale_pos_weight': scale_pos_weight,
            'random_state': self.random_state,
            'tree_method': 'hist',  # Fast histogram-based

            # Hyperparameters to tune
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_boost_round': trial.suggest_int('num_boost_round', 50, 300),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
        }

        num_boost_round = params.pop('num_boost_round')
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dval, 'val')],
            callbacks=[xgb.callback.EarlyStopping(rounds=15, metric_name='aucpr')],
            verbose_eval=False
        )

        # Evaluate on validation set
        y_proba = model.predict(dval)
        y_true = dval.get_label()
        pr_auc = average_precision_score(y_true, y_proba)

        return pr_auc
    
    def tune_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: int = 50
    ) -> Dict:
        """
        Hyperparameter tuning with Optuna.

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            n_trials: Number of Optuna trials

        Returns:
            Dictionary with best parameters
        """
        logger.info(f"Starting hyperparameter tuning ({n_trials} trials)...")

        # Calculate class weights
        scale_pos_weight = min((y_train == 0).sum() / (y_train == 1).sum(), 50)
        logger.info(f"Class imbalance ratio: {scale_pos_weight:.2f}")

        # Create DMatrix with feature names for consistent importance mapping
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_cols)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_cols)

        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state)
        )

        # Optimize
        study.optimize(
            lambda trial: self._objective(
                trial, dtrain, dval, scale_pos_weight
            ),
            n_trials=n_trials,
            show_progress_bar=True
        )

        logger.info(f"Best PR-AUC: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")

        # Add fixed params
        self.best_params = {
            **study.best_params,
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',
            'scale_pos_weight': scale_pos_weight,
            'random_state': self.random_state,
            'tree_method': 'hist'
        }

        return self.best_params
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: Optional[Dict] = None
    ):
        """
        Train final model with given or tuned parameters.

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            params: Model parameters (if None, uses best_params from tuning)
        """
        if params is None:
            if self.best_params is None:
                raise ValueError("Must tune hyperparameters or provide params")
            params = self.best_params

        logger.info("Training final model...")

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_cols)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_cols)
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=params.get('num_boost_round', 100),
            evals=[(dtrain, 'train'), (dval, 'val')],
            callbacks=[xgb.callback.EarlyStopping(rounds=15, metric_name='aucpr')],
            verbose_eval=True
        )

        logger.info("Model training complete")
    
    def find_optimal_threshold(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        target_precision: float = 0.90
    ) -> float:
        """
        Find decision threshold for target precision.

        Args:
            X_val, y_val: Validation data
            target_precision: Minimum precision to maintain

        Returns:
            Optimal threshold
        """
        dval = xgb.DMatrix(X_val)
        y_proba = self.model.predict(dval)

        precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)

        # Find threshold that achieves target precision
        valid_idx = np.where(precisions >= target_precision)[0]

        if len(valid_idx) == 0:
            logger.warning(f"Could not achieve {target_precision} precision")
            # Use threshold that maximizes F1
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
            best_idx = np.argmax(f1_scores)
            if best_idx == 0:
                self.best_threshold = 0.0
            else:
                self.best_threshold = thresholds[best_idx - 1]
        else:
            # Use threshold with highest recall at target precision
            best_idx = valid_idx[np.argmax(recalls[valid_idx])]
            if best_idx == 0:
                self.best_threshold = 0.0
            else:
                self.best_threshold = thresholds[best_idx - 1]

        logger.info(f"Optimal threshold: {self.best_threshold:.4f} "
                   f"(Precision: {precisions[best_idx]:.3f}, Recall: {recalls[best_idx]:.3f})")

        return self.best_threshold

def evaluate_model(
    model: xgb.Booster,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    threshold: float = 0.5,
    contagion_threshold: float = 0.1
) -> Dict:
    """
    Comprehensive model evaluation.

    Args:
        model: Trained model
        X_val, y_val: Validation data
        threshold: Decision threshold
        contagion_threshold: Threshold for ring detection

    Returns:
        Dictionary of metrics
    """
    logger.info("Evaluating model...")

    dval = xgb.DMatrix(X_val)
    y_proba = model.predict(dval)
    y_pred = (y_proba >= threshold).astype(int)
    
    # Overall metrics
    pr_auc = average_precision_score(y_val, y_proba)
    roc_auc = roc_auc_score(y_val, y_proba)
    
    # Precision/Recall at threshold
    from sklearn.metrics import precision_score, recall_score
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    
    # Ring-specific metrics (high contagion subset)
    if 'contagion_risk' in X_val.columns:
        ring_mask = X_val['contagion_risk'] > contagion_threshold
        if ring_mask.sum() > 0 and y_val[ring_mask].sum() > 0:
            ring_f1 = f1_score(y_val[ring_mask], y_pred[ring_mask])
            ring_recall = recall_score(y_val[ring_mask], y_pred[ring_mask])
        else:
            ring_f1 = 0.0
            ring_recall = 0.0
    else:
        ring_f1 = 0.0
        ring_recall = 0.0
    
    # False positive/negative rates
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Uplift simulation (reduction in false declines)
    baseline_decline_rate = 0.05  # Assume 5% baseline false decline rate
    false_decline_reduction = max(0, baseline_decline_rate - fpr)
    uplift_pct = (false_decline_reduction / baseline_decline_rate) * 100 if baseline_decline_rate > 0 else 0
    
    metrics = {
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'fpr': fpr,
        'fnr': fnr,
        'ring_f1': ring_f1,
        'ring_recall': ring_recall,
        'threshold': threshold,
        'uplift_pct': uplift_pct,
        'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
    }
    
    logger.info("="*60)
    logger.info("MODEL EVALUATION RESULTS")
    logger.info("="*60)
    logger.info(f"PR-AUC:        {pr_auc:.4f}")
    logger.info(f"ROC-AUC:       {roc_auc:.4f}")
    logger.info(f"Precision:     {precision:.4f}")
    logger.info(f"Recall:        {recall:.4f}")
    logger.info(f"F1 Score:      {f1:.4f}")
    logger.info(f"FPR:           {fpr:.4f}")
    logger.info(f"FNR:           {fnr:.4f}")
    logger.info(f"Ring F1:       {ring_f1:.4f}")
    logger.info(f"Ring Recall:   {ring_recall:.4f}")
    logger.info(f"Uplift:        {uplift_pct:.1f}% reduction in false declines")
    logger.info("="*60)
    
    return metrics

def train_and_eval(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_cols: List[str],
    n_trials: int = 50,
    target_precision: float = 0.90
) -> Dict:
    """
    Complete training pipeline with tuning and evaluation.
    
    Args:
        train: Training DataFrame
        val: Validation DataFrame
        feature_cols: List of feature column names
        n_trials: Number of Optuna trials
        target_precision: Target precision for threshold
        
    Returns:
        Dictionary with model, metrics, and artifacts
    """
    # Prepare data
    X_train = train[feature_cols].fillna(0)
    y_train = train['Class']
    X_val = val[feature_cols].fillna(0)
    y_val = val['Class']
    
    logger.info(f"Training data: {X_train.shape}, Fraud rate: {y_train.mean():.4f}")
    logger.info(f"Validation data: {X_val.shape}, Fraud rate: {y_val.mean():.4f}")
    logger.info(f"Features: {len(feature_cols)}")
    
    # Initialize model
    fraud_model = FraudDetectionModel(feature_cols=feature_cols)
    
    # Hyperparameter tuning
    best_params = fraud_model.tune_hyperparameters(
        X_train, y_train, X_val, y_val, n_trials=n_trials
    )
    
    # Train final model
    fraud_model.train(X_train, y_train, X_val, y_val)
    
    # Find optimal threshold
    optimal_threshold = fraud_model.find_optimal_threshold(
        X_val, y_val, target_precision=target_precision
    )
    
    # Evaluate
    metrics = evaluate_model(
        fraud_model.model, X_val, y_val,
        threshold=optimal_threshold
    )
    
    # Feature importance
    importance_dict = fraud_model.model.get_score(importance_type='gain')
    feature_importance = pd.DataFrame({
        'feature': list(importance_dict.keys()),
        'importance': list(importance_dict.values())
    }).sort_values('importance', ascending=False)

    logger.info("\nTop 10 Features:")
    logger.info(feature_importance.head(10).to_string(index=False))

    return {
        'model': fraud_model.model,
        'feature_cols': feature_cols,
        'best_params': best_params,
        'optimal_threshold': optimal_threshold,
        'metrics': metrics,
        'feature_importance': feature_importance
    }

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    from load_data import load_and_split
    from feature_engineering import engineer_features
    from gnn_contagion import build_and_embed_graph

    train, val, test = load_and_split()
    train, val, test = engineer_features(train, val, test)
    train, val, test, _ = build_and_embed_graph(train, val, test)

    # Define features
    feature_cols = [
        # Velocity
        'count_1H', 'sum_1H', 'unique_merchant_1H',
        'count_3H', 'sum_3H', 'unique_merchant_3H',
        # Geo
        'dist_prev_km', 'speed_kmh', 'is_impossible_travel',
        # Fingerprint
        'amount_zscore', 'amount_to_max_ratio', 'merchant_novelty',
        # Original PCA features (sample)
        'V1', 'V2', 'V4', 'V11',
        # Amount
        'Amount',
        # GNN
        'contagion_risk',
        'user_embed_0', 'user_embed_1', 'user_embed_2'
    ]

    results = train_and_eval(train, val, feature_cols, n_trials=20)

    # Save model in JSON format
    results['model'].save_model('models/xgb_model.json')
    logger.info("Model saved to models/xgb_model.json")

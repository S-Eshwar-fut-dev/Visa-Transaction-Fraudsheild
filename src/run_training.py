import os
import sys
import logging
from datetime import datetime
import yaml
import joblib
import pandas as pd

from load_data import load_and_split
from feature_engineering import engineer_features
from gnn_contagion import build_and_embed_graph
from train_xgboost import train_and_eval
from shap_explain import FraudExplainer

def setup_logging(log_path: str = 'logs/training.log'):
    """Setup logging configuration with file and console output."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def load_config(config_path: str = 'config.yaml') -> dict:
    """Load configuration from YAML file with defaults fallback."""
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(config_path):
        logger.warning(f"Config file {config_path} not found. Using default configuration.")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}

def main():
    # Setup logging first
    logger = setup_logging()
    
    logger.info("=" * 80)
    logger.info("Visa Fraudshield - MODEL TRAINING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load config
    config = load_config()

    try:
        # 1. Load and split data
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: DATA LOADING & AUGMENTATION")
        logger.info("=" * 80)
        
        train, val, test = load_and_split(
            path=config.get('data', {}).get('raw_path', 'data/raw/creditcard.csv'),
            train_ratio=config.get('splits', {}).get('train_ratio', 0.8),
            val_ratio=config.get('splits', {}).get('val_ratio', 0.1),
            random_seed=config.get('splits', {}).get('random_seed', 42)
        )
        
        logger.info(f"Data loaded: Train={len(train)}, Val={len(val)}, Test={len(test)}")
        logger.info(f"Fraud rate (train): {train['Class'].mean():.4%}")

        # 2. Feature engineering
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: FEATURE ENGINEERING")
        logger.info("=" * 80)
        
        train, val, test = engineer_features(
            train, val, test,
            windows=config.get('features', {}).get('rolling_windows', ['1H']),
            speed_threshold=config.get('features', {}).get('geo_speed_threshold_kmh', 1000)
        )
        
        logger.info("Feature engineering completed")

        # 3. GNN embeddings and contagion risk
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: GRAPH NEURAL NETWORK EMBEDDINGS & CONTAGION RISK")
        logger.info("=" * 80)
        
        embed_dim = config.get('gnn', {}).get('embed_dim', 12)
        cache_path = config.get('paths', {}).get('embeddings_cache', 'models/graph_embedder.pkl')
        
        train, val, test, embedder = build_and_embed_graph(
            train, val, test,
            embed_dim=embed_dim,
            cache_path=cache_path
        )
        
        logger.info(f"GNN features added (embed_dim={embed_dim})")

        # 4. Define feature columns
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: FEATURE SELECTION")
        logger.info("=" * 80)
        
        base_features = [
            # Velocity (adjust based on your engineer_features output)
            'count_1h', 'sum_1h', 'unique_merchant_1h',
            # Geo
            'dist_prev_km', 'is_impossible_travel',
            # Fingerprint
            'amount_zscore', 'merchant_novelty',
            # Original PCA + Amount
            'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
            'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
            'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',
            # GNN
            'contagion_risk'
        ]
        
        # Dynamically add embedding columns
        embed_cols = [f'user_embed_{i}' for i in range(embed_dim)]
        feature_cols = base_features + embed_cols
        
        # Filter to only existing columns
        available_features = [col for col in feature_cols if col in train.columns]
        missing_features = set(feature_cols) - set(available_features)
        
        if missing_features:
            logger.warning(f"Missing features (will be skipped): {missing_features}")
        
        logger.info(f"Final feature count: {len(available_features)}")

        # 5. Train model with Optuna tuning
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: MODEL TRAINING & HYPERPARAMETER TUNING")
        logger.info("=" * 80)
        
        results = train_and_eval(
            train, val,
            feature_cols=available_features,
            n_trials=config.get('model', {}).get('optuna_trials', 50),
            target_precision=config.get('thresholds', {}).get('target_precision', 0.90)
        )
        
        logger.info("Training completed successfully")

        # 6. Build SHAP explainer
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: BUILDING EXPLAINER")
        logger.info("=" * 80)
        
        explainer = FraudExplainer(results['model'], available_features)
        explainer_path = config.get('paths', {}).get('explainer_output', 'models/explainer_config.pkl')
        explainer.save(explainer_path)
        logger.info(f"Explainer saved to {explainer_path}")

        # 7. Save complete pipeline
        logger.info("\n" + "=" * 80)
        logger.info("STEP 7: SAVING FINAL PIPELINE")
        logger.info("=" * 80)
        
        os.makedirs('models', exist_ok=True)
        
        pipeline = {
            'model': results['model'],
            'feature_cols': available_features,
            'best_params': results.get('best_params', {}),
            'optimal_threshold': results.get('optimal_threshold'),
            'metrics': results.get('metrics', {}),
            'feature_importance': results.get('feature_importance'),
            'config_used': config,
            'timestamp': datetime.now().isoformat()
        }
        
        pipeline_path = config.get('paths', {}).get('model_output', 'models/pipeline.pkl')
        joblib.dump(pipeline, pipeline_path)
        joblib.dump(available_features, 'models/feature_cols.pkl')
        
        logger.info(f"Complete pipeline saved to {pipeline_path}")

        # 8. Final summary
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE - FINAL SUMMARY")
        logger.info("=" * 80)
        
        metrics = results.get('metrics', {})
        logger.info(f"PR-AUC:               {metrics.get('pr_auc', 0):.4f}")
        logger.info(f"ROC-AUC:              {metrics.get('roc_auc', 0):.4f}")
        logger.info(f"Precision:            {metrics.get('precision', 0):.4f}")
        logger.info(f"Recall:               {metrics.get('recall', 0):.4f}")
        logger.info(f"F1 Score:             {metrics.get('f1_score', 0):.4f}")
        logger.info(f"Ring F1:              {metrics.get('ring_f1', 0):.4f}")
        logger.info(f"Optimal Threshold:    {results.get('optimal_threshold', 0):.4f}")
        logger.info(f"False Decline Uplift: {metrics.get('uplift_pct', 0):.1f}%")
        logger.info(f"Best Trial Params:    {results.get('best_params', {})}")
        logger.info(f"\nPipeline:             {pipeline_path}")
        logger.info(f"End time:             {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        
        return pipeline

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    pipeline = main()
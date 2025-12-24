from load_data import load_and_split
from feature_engineering import engineer_features
from gnn_contagion import build_and_embed_graph
from train_xgboost import train_and_eval
from shap_explain import build_explainer, get_reason_codes
import joblib

if __name__ == "__main__":
    train, val, test = load_and_split()
    train, val, test = engineer_features(train, val, test)
    train = build_and_embed_graph(train)
    
    feature_cols = ['count_1h', 'sum_1h', 'unique_merchant_1h', 'dist_prev_km', 'is_impossible_travel',
                    'amount_zscore', 'merchant_novelty', 'V1', 'V2', 'V4', 'V11', 'Amount', 'contagion_risk']
    results = train_and_eval(train, val, feature_cols)
    
    explainer = build_explainer(results['model'], feature_cols)
    pipeline = {'model': results['model'], 'explainer': explainer, 'features': feature_cols,
                'metrics': {'pr_auc': results['pr_auc'], 'recall_90': results['recall_at_90prec']}}
    joblib.dump(pipeline, 'models/pipeline.pkl')
    
    print("Pipeline saved. Metrics:", results)
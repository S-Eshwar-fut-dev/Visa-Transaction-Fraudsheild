# scripts/eval_and_report.sh
# Generate comprehensive evaluation report

set -e

echo "=========================================="
echo "Evaluation Report Generator"
echo "=========================================="
echo ""

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check if model exists
if [ ! -f "models/pipeline.pkl" ]; then
    echo "Error: Model not found. Run ./scripts/run_pipeline.sh first"
    exit 1
fi

# Create reports directory
mkdir -p reports

echo "Generating evaluation metrics..."

# Run evaluation Python script
python3 << 'EOF'
import sys
sys.path.insert(0, 'src')

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc,
    average_precision_score, roc_auc_score,
    precision_score, recall_score, f1_score,
    confusion_matrix
)
import xgboost as xgb
import json
from datetime import datetime

print("Loading model and data...")

# Load model
pipeline = joblib.load('models/pipeline.pkl')
model = pipeline['model']
feature_cols = pipeline['feature_cols']
threshold = pipeline['optimal_threshold']

# Load validation data
val_raw = pd.read_csv('data/processed/val_raw.csv')

# Load feature-engineered data if available
try:
    val = pd.read_csv('data/processed/val_features.csv')
except:
    print("Warning: Feature-engineered val not found, using raw")
    val = val_raw

# Ensure all features exist
for col in feature_cols:
    if col not in val.columns:
        val[col] = 0

X_val = val[feature_cols].fillna(0)
y_val = val['Class'] if 'Class' in val.columns else np.zeros(len(val))

# Predict
dval = xgb.DMatrix(X_val)
y_proba = model.predict(dval)
y_pred = (y_proba >= threshold).astype(int)

print("\nComputing metrics...")

# Overall metrics
pr_auc = average_precision_score(y_val, y_proba) if y_val.sum() > 0 else 0
roc_auc = roc_auc_score(y_val, y_proba) if y_val.sum() > 0 else 0
precision = precision_score(y_val, y_pred, zero_division=0)
recall = recall_score(y_val, y_pred, zero_division=0)
f1 = f1_score(y_val, y_pred, zero_division=0)

# Confusion matrix
tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

# Ring metrics (if contagion_risk available)
ring_f1 = 0.0
ring_recall = 0.0
if 'contagion_risk' in val.columns:
    ring_mask = val['contagion_risk'] > 0.05
    if ring_mask.sum() > 0 and y_val[ring_mask].sum() > 0:
        ring_f1 = f1_score(y_val[ring_mask], y_pred[ring_mask], zero_division=0)
        ring_recall = recall_score(y_val[ring_mask], y_pred[ring_mask], zero_division=0)

# False decline uplift
baseline_fpr = 0.05  # 5% baseline
uplift_pct = max(0, (baseline_fpr - fpr) / baseline_fpr * 100)

# Revenue calculation
monthly_txs = 1_000_000
avg_tx = 75
legit_txs = monthly_txs * 0.998
false_declines_baseline = legit_txs * baseline_fpr
false_declines_model = legit_txs * fpr
recovered = max(0, false_declines_baseline - false_declines_model)
monthly_revenue = recovered * avg_tx
annual_revenue = monthly_revenue * 12

# Per-merchant analysis
merchant_stats = None
if 'merchant_id' in val.columns:
    merchant_df = val.groupby('merchant_id').agg({
        'Class': ['sum', 'count', 'mean']
    }).reset_index()
    merchant_df.columns = ['merchant_id', 'fraud_count', 'tx_count', 'fraud_rate']
    merchant_stats = merchant_df.nlargest(10, 'fraud_rate').to_dict('records')

# Compile results
results = {
    'timestamp': datetime.now().isoformat(),
    'model_info': {
        'features': len(feature_cols),
        'threshold': float(threshold),
        'hyperparameters': pipeline.get('best_params', {})
    },
    'overall_metrics': {
        'pr_auc': float(pr_auc),
        'roc_auc': float(roc_auc),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'fpr': float(fpr),
        'fnr': float(fnr)
    },
    'ring_metrics': {
        'ring_f1': float(ring_f1),
        'ring_recall': float(ring_recall)
    },
    'confusion_matrix': {
        'true_negative': int(tn),
        'false_positive': int(fp),
        'false_negative': int(fn),
        'true_positive': int(tp)
    },
    'business_impact': {
        'uplift_pct': float(uplift_pct),
        'monthly_revenue_uplift': float(monthly_revenue),
        'annual_revenue_uplift': float(annual_revenue),
        'recovered_txs_monthly': float(recovered),
        'assumptions': {
            'monthly_txs': monthly_txs,
            'avg_tx_value': avg_tx,
            'baseline_fpr': baseline_fpr
        }
    },
    'top_merchants': merchant_stats
}

# Save JSON
with open('reports/metrics.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("EVALUATION SUMMARY")
print("="*60)
print(f"PR-AUC:               {pr_auc:.4f}")
print(f"ROC-AUC:              {roc_auc:.4f}")
print(f"Precision:            {precision:.4f}")
print(f"Recall:               {recall:.4f}")
print(f"F1 Score:             {f1:.4f}")
print(f"Ring F1:              {ring_f1:.4f}")
print(f"False Positive Rate:  {fpr:.4f}")
print(f"False Decline Uplift: {uplift_pct:.1f}%")
print(f"\nBusiness Impact:")
print(f"  Monthly Revenue:    ${monthly_revenue:,.0f}")
print(f"  Annual Revenue:     ${annual_revenue:,.0f}")
print(f"  Recovered TXs/mo:   {recovered:,.0f}")
print("="*60)
print(f"\nReport saved: reports/metrics.json")

EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Evaluation complete!"
    echo ""
    echo "Generated files:"
    ls -lh reports/
    echo ""
else
    echo "✗ Evaluation failed"
    exit 1
fi
# pages/2_Explain_Deep_Dive.py
import streamlit as st
import pandas as pd
import numpy as np
import shap
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.data_loader import load_model_and_data, predict_batch

st.set_page_config(page_title="Explain Deep Dive", page_icon="🔬", layout="wide")

st.title("🔬 SHAP Explainability Deep Dive")

try:
    model, feature_cols, val_data = load_model_and_data()
    
    # Select transaction
    st.sidebar.header("🎯 Select Transaction")
    
    sample_txs = val_data.sample(min(50, len(val_data)))
    X_sample = sample_txs[feature_cols].fillna(0)
    preds_sample = predict_batch(model, X_sample)
    
    tx_options = [f"TX-{i:03d} (Risk: {p:.3f})" for i, p in enumerate(preds_sample)]
    selected = st.sidebar.selectbox("Transaction", tx_options)
    
    tx_idx = int(selected.split("-")[1].split(" ")[0])
    
    # Get transaction data
    tx_data = sample_txs.iloc[tx_idx]
    X_tx = X_sample.iloc[[tx_idx]]
    risk_score = preds_sample[tx_idx]
    
    # Display transaction
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Transaction Details")
        st.metric("Risk Score", f"{risk_score:.3f}")
        st.metric("Amount", f"${tx_data['Amount']:.2f}")
        st.metric("Actual Fraud", "Yes" if tx_data['Class'] == 1 else "No")
        
        if 'contagion_risk' in tx_data:
            st.metric("Contagion Risk", f"{tx_data['contagion_risk']:.3f}")
    
    with col2:
        # SHAP Explanation
        st.markdown("### SHAP Force Plot")
        
        with st.spinner("Computing SHAP values..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_tx)
            
            # Handle multi-output
            if isinstance(shap_values.values, np.ndarray) and shap_values.values.ndim == 3:
                vals = shap_values.values[0, :, 1]
            else:
                vals = shap_values.values[0]
            
            # Create waterfall
            fig = go.Figure()
            
            # Sort by absolute contribution
            feature_contribs = list(zip(feature_cols, vals))
            feature_contribs.sort(key=lambda x: abs(x[1]), reverse=True)
            
            top_features = feature_contribs[:10]
            features, contribs = zip(*top_features)
            
            colors = ['#ef4444' if c > 0 else '#10b981' for c in contribs]
            
            fig.add_trace(go.Bar(
                y=list(features),
                x=list(contribs),
                orientation='h',
                marker=dict(color=colors),
                text=[f"{c:+.4f}" for c in contribs],
                textposition='outside'
            ))
            
            fig.update_layout(
                title="Top 10 Feature Contributions",
                xaxis_title="SHAP Value",
                yaxis_title="Feature",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(30, 41, 59, 0.5)',
                font=dict(color='#e2e8f0'),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Reason Codes
    st.markdown("### 🏷️ Visa-Style Reason Codes")
    
    REASON_CODES = {
        'amount_zscore': 'R01: Amount Anomaly',
        'contagion_risk': 'R02: Fraud Ring Association',
        'is_impossible_travel': 'R03: Geographic Anomaly',
        'count_1H': 'R04: Velocity Burst',
        'merchant_novelty': 'R06: New Merchant Pattern'
    }
    
    top_3 = [(f, c) for f, c in feature_contribs[:3] if c > 0]
    
    if top_3:
        for feature, contrib in top_3:
            code = REASON_CODES.get(feature, f"R99: {feature}")
            st.info(f"**{code}** (Impact: +{contrib:.4f})")
    else:
        st.success("✅ Low risk - no major fraud indicators")

except Exception as e:
    st.error(f"Error: {str(e)}")
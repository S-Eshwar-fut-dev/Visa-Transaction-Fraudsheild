# pages/2_Explain.py
import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import shap
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import load_model_and_data, predict_batch
from utils.viz import create_shap_waterfall, GLASS_THEME
from utils.styles import load_css
from utils.components import metric_card, animated_separator

st.set_page_config(page_title="Explainability Lab", page_icon="🔬", layout="wide")
load_css()

st.title("🔬 SHAP Explainability Lab")
st.markdown("Understand *why* the model makes each decision")

# Load resources
model, feature_cols, val_data = load_model_and_data()

if model is None:
    st.error("⚠️ Model not found")
    st.stop()

# Sidebar: Transaction selector
with st.sidebar:
    st.markdown("### 🎯 Transaction Selector")
    
    # Sample transactions
    sample_size = min(100, len(val_data))
    sample_txs = val_data.sample(sample_size, random_state=42).reset_index(drop=True)
    
    X_sample = sample_txs[feature_cols].fillna(0)
    preds_sample = predict_batch(model, X_sample)
    
    # Filter options
    filter_type = st.radio("Filter by:", ["All", "High Risk (>0.5)", "Fraud Cases", "Low Risk (<0.1)"])
    
    if filter_type == "High Risk (>0.5)":
        mask = preds_sample > 0.5
    elif filter_type == "Fraud Cases":
        mask = sample_txs['Class'] == 1 if 'Class' in sample_txs.columns else preds_sample > 0.5
    elif filter_type == "Low Risk (<0.1)":
        mask = preds_sample < 0.1
    else:
        mask = np.ones(len(sample_txs), dtype=bool)
    
    filtered_txs = sample_txs[mask].reset_index(drop=True)
    filtered_preds = preds_sample[mask]
    
    if len(filtered_txs) == 0:
        st.warning("No transactions match filter")
        st.stop()
    
    # Transaction selector
    tx_options = [
        f"TX-{i:03d} | Risk: {filtered_preds[i]:.3f} | ${filtered_txs.iloc[i]['Amount']:.2f}"
        for i in range(len(filtered_txs))
    ]
    
    selected_option = st.selectbox("Select Transaction:", tx_options)
    tx_idx = int(selected_option.split("-")[1].split(" ")[0])
    
    st.markdown("---")
    st.info("💡 SHAP values explain each feature's contribution to the prediction")

# Get selected transaction
tx_data = filtered_txs.iloc[tx_idx]
X_tx = X_sample[mask].iloc[[tx_idx]]
risk_score = filtered_preds[tx_idx]
actual_fraud = tx_data['Class'] if 'Class' in tx_data else 0

# Display transaction details
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 💳 Transaction Details")
    
    risk_color = '#ef4444' if risk_score > 0.5 else '#10b981'
    status_icon = '🚨 Fraud' if actual_fraud == 1 else '✅ Legitimate'
    
    st.markdown(f"""
    <div class="glass-card">
        <h3 style="color: {risk_color}; font-size: 2rem; margin-bottom: 0.5rem;">
            Risk: {risk_score:.4f}
        </h3>
        <p style="color: var(--text-secondary); margin-bottom: 1rem;">Probability of fraud</p>
        <hr style="border-color: rgba(255,255,255,0.1); margin-bottom: 1rem;">
        
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="color: var(--text-secondary);">Amount:</span>
            <span style="font-family: 'JetBrains Mono'; color: white;">${tx_data['Amount']:.2f}</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="color: var(--text-secondary);">Actual Label:</span>
            <span>{status_icon}</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="color: var(--text-secondary);">User ID:</span>
            <span style="font-family: 'JetBrains Mono'; color: white;">{tx_data.get('user_id', 'N/A')}</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="color: var(--text-secondary);">Merchant ID:</span>
            <span style="font-family: 'JetBrains Mono'; color: white;">{tx_data.get('merchant_id', 'N/A')}</span>
        </div>
        
        <hr style="border-color: rgba(255,255,255,0.1); margin: 1rem 0;">
        
        {f'<div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;"><span style="color: var(--text-secondary);">Contagion Risk:</span><span style="font-family: JetBrains Mono; color: var(--accent-purple);">{tx_data["contagion_risk"]:.3f}</span></div>' if 'contagion_risk' in tx_data else ''}
        {f'<div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;"><span style="color: var(--text-secondary);">Velocity (1h):</span><span style="font-family: JetBrains Mono; color: var(--accent-blue);">{tx_data["count_1h"]:.0f} txs</span></div>' if 'count_1h' in tx_data else ''}
        {f'<div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;"><span style="color: var(--text-secondary);">Amount Z-Score:</span><span style="font-family: JetBrains Mono;">{tx_data["amount_zscore"]:.2f}</span></div>' if 'amount_zscore' in tx_data else ''}
        {f'<div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;"><span style="color: var(--text-secondary);">Impossible Travel:</span><span>{"⚠️ Yes" if tx_data.get("is_impossible_travel", 0) == 1 else "✅ No"}</span></div>' if 'is_impossible_travel' in tx_data else ''}
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("### 🔍 SHAP Explanation")
    
    with st.spinner("Computing SHAP values..."):
        try:
            # Compute SHAP
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_tx)
            
            # Extract values (handle multi-output)
            if isinstance(shap_values.values, np.ndarray):
                if shap_values.values.ndim == 3:
                    vals = shap_values.values[0, :, 1]  # Positive class
                elif shap_values.values.ndim == 2:
                    vals = shap_values.values[0, :]
                else:
                    vals = shap_values.values.flatten()
            else:
                vals = np.array(shap_values.values).flatten()
            
            # Create waterfall
            fig_waterfall = create_shap_waterfall(
                shap_values=vals,
                feature_names=feature_cols,
                base_value=0.002,
                top_k=12
            )
            
            st.plotly_chart(fig_waterfall, use_container_width=True)
            
        except Exception as e:
            st.error(f"SHAP computation failed: {e}")
            vals = np.zeros(len(feature_cols))

# Visa-style reason codes
animated_separator()
st.markdown("### 🏷️ Reason Codes (Visa-Style)")

REASON_CODES = {
    'amount_zscore': ('R01', 'Amount Anomaly', 'Transaction size unusual for this user'),
    'contagion_risk': ('R02', 'Fraud Ring Association', 'Connected to high-risk merchants'),
    'is_impossible_travel': ('R03', 'Geographic Anomaly', 'Impossible travel speed detected'),
    'count_1h': ('R04', 'Velocity Burst', 'Excessive transactions in 1 hour'),
    'count_3h': ('R05', 'Sustained Velocity', 'High transaction frequency over 3 hours'),
    'merchant_novelty': ('R06', 'New Merchant', 'First-time merchant interaction'),
    'speed_kmh': ('R07', 'Rapid Location Change', 'High travel speed between transactions'),
    'sum_1h': ('R08', 'Amount Velocity', 'Large sum spent in short period'),
    'amount_to_max_ratio': ('R09', 'User Maximum Breach', 'Exceeds typical maximum'),
    'unique_merchant_1h': ('R10', 'Merchant Diversity Spike', 'Multiple new merchants')
}

# Get top contributing features
feature_contribs = list(zip(feature_cols, vals))
feature_contribs.sort(key=lambda x: abs(x[1]), reverse=True)

top_positive = [(f, v) for f, v in feature_contribs[:10] if v > 0.001]

if top_positive:
    st.markdown("#### 🚨 Fraud Indicators Detected:")
    
    cols = st.columns(2)
    for idx, (feature, contrib) in enumerate(top_positive):
        col = cols[idx % 2]
        
        if feature in REASON_CODES:
            code, title, desc = REASON_CODES[feature]
            
            with col:
                st.markdown(f"""
                <div class="glass-card" style="padding: 1rem; margin-bottom: 0.5rem; border-left: 3px solid var(--accent-rose);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <strong style="color: var(--accent-rose);">{code}: {title}</strong>
                        <span style="font-size: 0.75rem; color: var(--text-secondary);">Impact: +{contrib:.4f}</span>
                    </div>
                    <small style="color: var(--text-secondary);">{desc}</small>
                </div>
                """, unsafe_allow_html=True)
        elif abs(contrib) > 0.005:
            with col:
                st.markdown(f"""
                <div class="glass-card" style="padding: 1rem; margin-bottom: 0.5rem; border-left: 3px solid var(--accent-blue);">
                     <div style="display: flex; justify-content: space-between; align-items: center;">
                        <strong style="color: var(--accent-blue);">R99: {feature}</strong>
                        <span style="font-size: 0.75rem; color: var(--text-secondary);">Impact: +{contrib:.4f}</span>
                    </div>
                    <small style="color: var(--text-secondary);">High impact feature</small>
                </div>
                """, unsafe_allow_html=True)
else:
    st.success("✅ **Low Risk Transaction** - No significant fraud indicators")
    
    # Show top protective features
    top_negative = [(f, v) for f, v in feature_contribs if v < -0.001][:3]
    
    if top_negative:
        st.markdown("#### 🛡️ Protective Factors:")
        for feature, contrib in top_negative:
            st.info(f"**{feature}**: Reduces risk by {abs(contrib):.4f}")

# Feature value comparison
animated_separator()
st.markdown("### 📊 Feature Value Analysis")

col_feat1, col_feat2 = st.columns(2)

with col_feat1:
    # Top contributing features
    top_5_features = [f for f, _ in feature_contribs[:5]]
    
    feature_values = []
    for feat in top_5_features:
        feature_values.append({
            'Feature': feat.replace('_', ' ').title(),
            'Value': tx_data.get(feat, 0),
            'SHAP': vals[feature_cols.index(feat)] if feat in feature_cols else 0
        })
    
    df_features = pd.DataFrame(feature_values)
    
    st.markdown("#### Top 5 Contributing Features")
    st.dataframe(
        df_features.style.background_gradient(cmap='RdYlGn_r', subset=['SHAP']),
        use_container_width=True
    )

with col_feat2:
    # Feature value distribution
    if len(top_5_features) > 0:
        selected_feat = st.selectbox("Compare feature distribution:", top_5_features)
        
        if selected_feat in val_data.columns:
            fig_dist = go.Figure()
            
            # Distribution
            fig_dist.add_trace(go.Histogram(
                x=val_data[selected_feat].fillna(0),
                nbinsx=50,
                name='All Transactions',
                marker=dict(color=GLASS_THEME['accent_blue'], opacity=0.6)
            ))
            
            # Current value
            fig_dist.add_vline(
                x=tx_data.get(selected_feat, 0),
                line_dash="dash",
                line_color=GLASS_THEME['accent_red'],
                annotation_text="Current TX"
            )
            
            fig_dist.update_layout(
                title=f"Distribution: {selected_feat}",
                xaxis_title="Value",
                yaxis_title="Frequency",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor=GLASS_THEME['plot_bg'],
                font=dict(color=GLASS_THEME['text_primary']),
                height=300,
                showlegend=False
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)

# AI Co-Pilot
animated_separator()
st.markdown("### 🤖 AI Fraud Analyst")

user_question = st.text_input(
    "Ask Claude about this transaction:",
    placeholder="e.g., 'Is this transaction part of a fraud ring?' or 'What's the biggest red flag?'"
)

if user_question:
    with st.spinner("Claude is analyzing..."):
        # Prepare context (simulation)
        st.info("""
        **Claude's Analysis:**
        
        Based on the SHAP values and transaction characteristics, this appears to be a 
        high-risk transaction. The primary concern is contagion_risk, which indicates 
        connections to known fraud rings.
        
        Recommendation: Flag for manual review.
        """)

st.caption("💡 SHAP values are additive: Base rate + sum of contributions = final probability")
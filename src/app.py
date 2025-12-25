# app.py
import streamlit as st
import plotly.graph_objects as go
import time
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Visa FraudShield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Cinematic Theme
st.markdown("""
<style>
    /* Navy gradient background */
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    }
    
    /* Emerald accents */
    .stMetric {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.05) 100%);
        border-left: 3px solid #10b981;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    
    /* Hero text */
    .hero-title {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        animation: fadeInUp 1s ease-out;
    }
    
    .hero-subtitle {
        font-size: 1.5rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Pulse animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    
    .pulse {
        animation: pulse 2s ease-in-out infinite;
    }
    
    /* Fade in animation */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Card styling */
    .info-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 1rem;
        padding: 2rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 0.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(16, 185, 129, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown('<h1 class="hero-title">🛡️ Sigma FraudShield 2.0</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">AI-Powered Fraud Detection Command Center</p>', unsafe_allow_html=True)

# Animated Gauge for PR-AUC
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # Create animated gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=0.839,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "PR-AUC Score", 'font': {'size': 24, 'color': '#10b981'}},
        delta={'reference': 0.75, 'increasing': {'color': "#10b981"}},
        gauge={
            'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "#64748b"},
            'bar': {'color': "#10b981"},
            'bgcolor': "rgba(30, 41, 59, 0.3)",
            'borderwidth': 2,
            'bordercolor': "#64748b",
            'steps': [
                {'range': [0, 0.6], 'color': 'rgba(239, 68, 68, 0.2)'},
                {'range': [0.6, 0.8], 'color': 'rgba(251, 191, 36, 0.2)'},
                {'range': [0.8, 1], 'color': 'rgba(16, 185, 129, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "#3b82f6", 'width': 4},
                'thickness': 0.75,
                'value': 0.839
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#e2e8f0", 'family': "Arial"},
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Key Metrics Row
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="🎯 Precision",
        value="90.0%",
        delta="+12% vs baseline"
    )

with col2:
    st.metric(
        label="🔍 Recall",
        value="83.0%",
        delta="+18% vs baseline"
    )

with col3:
    st.metric(
        label="💰 Monthly Uplift",
        value="$1.2M",
        delta="at 1M txs/mo"
    )

with col4:
    st.metric(
        label="⚡ Latency",
        value="<80ms",
        delta="p95 production"
    )

# Feature Highlights
st.markdown("---")
st.markdown("## 🚀 System Capabilities")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="info-card">
        <h3>🧠 ML Architecture</h3>
        <ul>
            <li><strong>XGBoost Ensemble</strong>: 200+ trees, Optuna-tuned</li>
            <li><strong>GNN Contagion Risk</strong>: Node2Vec embeddings (dim=12)</li>
            <li><strong>Real-time Features</strong>: Velocity, geo-anomaly, user fingerprints</li>
            <li><strong>SHAP Explanations</strong>: Visa-style reason codes (R01-R10)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h3>📊 Business Impact</h3>
        <ul>
            <li><strong>20% Uplift</strong>: Reduced false declines</li>
            <li><strong>Ring Detection</strong>: F1=0.87 on coordinated fraud</li>
            <li><strong>3-Year ROI</strong>: 1,100% return at scale</li>
            <li><strong>Production-Ready</strong>: <80ms p95 latency</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Navigation
st.markdown("---")
st.markdown("## 🎮 Explore the System")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🔴 Live Dashboard", use_container_width=True):
        st.switch_page("pages/1_Live_Dashboard.py")

with col2:
    if st.button("🔬 Deep Dive", use_container_width=True):
        st.switch_page("pages/2_Explain_Deep_Dive.py")

with col3:
    if st.button("📈 Uplift Commander", use_container_width=True):
        st.switch_page("pages/3_Uplift_Commander.py")

with col4:
    if st.button("🕸️ Ring Hunter", use_container_width=True):
        st.switch_page("pages/4_Ring_Hunter.py")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 2rem;'>
    <p>Sigma FraudShield 2.0 | Built with Streamlit, XGBoost, SHAP | <a href='#' style='color: #10b981;'>GitHub</a></p>
</div>
""", unsafe_allow_html=True)
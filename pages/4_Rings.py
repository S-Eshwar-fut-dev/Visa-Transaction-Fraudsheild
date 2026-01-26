# pages/4_Rings.py
import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import load_model_and_data, predict_batch
from utils.viz import create_3d_scatter, create_network_graph, GLASS_THEME
from utils.styles import load_css
from utils.components import metric_card, animated_separator

st.set_page_config(page_title="Ring Hunter", page_icon="🕸️", layout="wide")
load_css()

st.title("🕸️ Fraud Ring Hunter")
st.markdown("Detect coordinated fraud patterns using graph analysis")

# Load resources
model, feature_cols, val_data = load_model_and_data()

if model is None:
    st.error("⚠️ Model not found")
    st.stop()

# Sidebar controls
with st.sidebar:
    st.markdown("### 🎯 Detection Parameters")
    
    contagion_threshold = st.slider(
        "Contagion Risk Threshold",
        min_value=0.0,
        max_value=0.5,
        value=0.1,
        step=0.01,
        help="Minimum contagion risk to flag potential ring members"
    )
    
    min_connections = st.slider(
        "Min Connections",
        min_value=2,
        max_value=10,
        value=3,
        help="Minimum shared merchants to consider a ring"
    )
    
    sample_size = st.slider(
        "Sample Size",
        min_value=100,
        max_value=2000,
        value=500,
        step=100,
        help="Number of transactions to analyze"
    )
    
    st.markdown("---")
    st.info("💡 Higher contagion = stronger connection to fraud networks")

# Filter high-risk transactions
if 'contagion_risk' in val_data.columns:
    high_risk = val_data[val_data['contagion_risk'] > contagion_threshold].copy()
else:
    # Fallback: use predictions
    X_val = val_data[feature_cols].fillna(0)
    preds = predict_batch(model, X_val)
    val_data['risk_score'] = preds
    high_risk = val_data[val_data['risk_score'] > 0.5].copy()

# Sample for visualization
sample_data = val_data.sample(min(sample_size, len(val_data)), random_state=42).copy()
X_sample = sample_data[feature_cols].fillna(0)
sample_preds = predict_batch(model, X_sample)

# Metrics
st.markdown("### 📊 Ring Detection Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    metric_card(
        "Potential Ring Members",
        f"{len(high_risk)}",
        f"{len(high_risk)/len(val_data)*100:.2f}% of total",
        delta_color="var(--accent-rose)"
    )

with col2:
    unique_users = high_risk['user_id'].nunique() if 'user_id' in high_risk.columns else 0
    metric_card("Unique Users", f"{unique_users}", "Accounts", delta_color="normal")

with col3:
    unique_merchants = high_risk['merchant_id'].nunique() if 'merchant_id' in high_risk.columns else 0
    metric_card("Flagged Merchants", f"{unique_merchants}", "Risk Points", delta_color="normal")

with col4:
    avg_contagion = high_risk['contagion_risk'].mean() if 'contagion_risk' in high_risk.columns else 0
    metric_card("Avg Contagion Risk", f"{avg_contagion:.3f}", "High", delta_color="var(--accent-rose)")

# Alert for detected rings
if len(high_risk) > 10:
    st.markdown(f"""
    <div class="glass-card" style="box-shadow: 0 0 20px rgba(239, 68, 68, 0.4); border-color: rgba(239, 68, 68, 0.5);">
        <h3 style="color: #ef4444; margin-top: 0; text-shadow: 0 0 10px rgba(239, 68, 68, 0.5);">🚨 Coordinated Fraud Detected</h3>
        <p><strong>{len(high_risk)}</strong> transactions flagged with high contagion risk.</p>
        <p>This indicates potential <strong>fraud ring activity</strong> - multiple users coordinating with high-risk merchants.</p>
        <p><em>Recommended Action:</em> Investigate user-merchant networks, freeze suspicious accounts, alert fraud ops team.</p>
    </div>
    """, unsafe_allow_html=True)

animated_separator()

# 3D Feature Space
st.markdown("### 🌌 3D Transaction Space")
st.caption("Visualize fraud patterns in high-dimensional feature space")

x_feat = st.selectbox(
    "X-axis:",
    ['amount_zscore', 'Amount', 'contagion_risk', 'count_1h'] if 'contagion_risk' in sample_data.columns else ['Amount', 'V1', 'V2'],
    index=0
)

y_feat = st.selectbox(
    "Y-axis:",
    ['contagion_risk', 'count_1h', 'amount_zscore', 'speed_kmh'] if 'contagion_risk' in sample_data.columns else ['V1', 'V2', 'V3'],
    index=0
)

z_feat = st.selectbox(
    "Z-axis:",
    ['count_1h', 'sum_1h', 'unique_merchant_1h', 'dist_prev_km'] if 'count_1h' in sample_data.columns else ['V4', 'V11', 'Amount'],
    index=0
)

fig_3d = create_3d_scatter(
    sample_data,
    sample_preds,
    x_col=x_feat,
    y_col=y_feat,
    z_col=z_feat
)

st.plotly_chart(fig_3d, use_container_width=True)

animated_separator()

# Network graph
st.markdown("### 🕸️ User-Merchant Network")

if 'user_id' in high_risk.columns and 'merchant_id' in high_risk.columns:
    # Build network from high-risk transactions
    network_sample = high_risk.head(min(50, len(high_risk))).copy()
    
    edges = []
    fraud_nodes = set()
    
    for _, row in network_sample.iterrows():
        user = f"U{row['user_id']}"
        merchant = f"M{row['merchant_id']}"
        amount = row['Amount']
        
        edges.append((user, merchant, amount))
        
        if row.get('Class', 0) == 1 or (row.get('contagion_risk', 0) > contagion_threshold):
            fraud_nodes.add(user)
    
    fig_network = create_network_graph(edges, fraud_nodes)
    st.plotly_chart(fig_network, use_container_width=True)
    
    st.caption("🔴 Red nodes = High-risk users | 🔵 Blue nodes = Users | 🟢 Green nodes = Merchants")
else:
    st.info("Network visualization requires user_id and merchant_id columns")

# Ring details table
animated_separator()
st.markdown("### 📋 High-Risk Transaction Details")

if len(high_risk) > 0:
    display_cols = ['Time', 'user_id', 'merchant_id', 'Amount', 'contagion_risk', 'Class']
    display_cols = [c for c in display_cols if c in high_risk.columns]
    
    display_df = high_risk[display_cols].head(50).copy()
    
    if 'Time' in display_df.columns:
        display_df['Time'] = pd.to_datetime(display_df['Time']).dt.strftime('%Y-%m-%d %H:%M')
    
    st.dataframe(
        display_df.style.background_gradient(cmap='Reds', subset=['contagion_risk'] if 'contagion_risk' in display_df.columns else []),
        use_container_width=True,
        height=400
    )
    
    # Download button
    csv = high_risk[display_cols].to_csv(index=False)
    st.download_button(
        label="📥 Download Full Ring Data (CSV)",
        data=csv,
        file_name="fraud_ring_transactions.csv",
        mime="text/csv"
    )
else:
    st.success("✅ No high-risk rings detected with current threshold")

# Geographic distribution (if available)
animated_separator()
st.markdown("### 🗺️ Geographic Distribution")

if 'V4' in sample_data.columns and 'V11' in sample_data.columns:
    # Use V4 as proxy for latitude, V11 for longitude
    fig_geo = go.Figure()
    
    # Legitimate transactions
    legit = sample_data[sample_data['Class'] == 0] if 'Class' in sample_data.columns else sample_data[sample_preds < 0.5]
    fig_geo.add_trace(go.Scattergeo(
        lon=legit['V11'],
        lat=legit['V4'],
        mode='markers',
        marker=dict(
            size=4,
            color=GLASS_THEME['accent_emerald'],
            opacity=0.4,
            line=dict(width=0.5, color='white')
        ),
        name='Legitimate',
        hovertemplate='Lat: %{lat:.2f}<br>Lon: %{lon:.2f}<extra></extra>'
    ))
    
    # Fraud transactions
    fraud = sample_data[sample_data['Class'] == 1] if 'Class' in sample_data.columns else sample_data[sample_preds > 0.5]
    if len(fraud) > 0:
        fig_geo.add_trace(go.Scattergeo(
            lon=fraud['V11'],
            lat=fraud['V4'],
            mode='markers',
            marker=dict(
                size=8,
                color=GLASS_THEME['accent_red'],
                opacity=0.8,
                symbol='diamond',
                line=dict(width=1, color='white')
            ),
            name='Fraud',
            hovertemplate='<b>FRAUD</b><br>Lat: %{lat:.2f}<br>Lon: %{lon:.2f}<extra></extra>'
        ))
    
    fig_geo.update_layout(
        geo=dict(
            scope='usa',
            projection_type='albers usa',
            showland=True,
            landcolor='rgb(20, 30, 48)',
            coastlinecolor='rgba(255, 255, 255, 0.2)',
            bgcolor=GLASS_THEME['plot_bg']
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=GLASS_THEME['text_primary']),
        height=500,
        showlegend=True,
        legend=dict(
            bgcolor='rgba(30, 41, 59, 0.8)',
            bordercolor=GLASS_THEME['card_border'],
            borderwidth=1
        )
    )
    
    st.plotly_chart(fig_geo, use_container_width=True)
else:
    st.info("Geographic visualization requires location features (V4, V11)")

# Temporal patterns
animated_separator()
st.markdown("### ⏰ Temporal Fraud Patterns")

if 'Time' in sample_data.columns:
    sample_data['Hour'] = pd.to_datetime(sample_data['Time']).dt.hour
    
    hour_fraud = sample_data.groupby('Hour').agg({
        'Class': 'sum' if 'Class' in sample_data.columns else lambda x: (sample_preds > 0.5).sum()
    }).reset_index()
    
    fig_temporal = go.Figure()
    
    fig_temporal.add_trace(go.Bar(
        x=hour_fraud['Hour'],
        y=hour_fraud['Class'],
        marker=dict(
            color=hour_fraud['Class'],
            colorscale='Reds',
            line=dict(color='rgba(255, 255, 255, 0.3)', width=1)
        ),
        hovertemplate='Hour: %{x}:00<br>Fraud Count: %{y}<extra></extra>'
    ))
    
    fig_temporal.update_layout(
        xaxis=dict(
            title="Hour of Day",
            tickmode='linear',
            tick0=0,
            dtick=2,
            gridcolor=GLASS_THEME['grid_color']
        ),
        yaxis=dict(
            title="Fraud Count",
            gridcolor=GLASS_THEME['grid_color']
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=GLASS_THEME['plot_bg'],
        font=dict(color=GLASS_THEME['text_primary']),
        height=350,
        showlegend=False
    )
    
    st.plotly_chart(fig_temporal, use_container_width=True)
    
    if not hour_fraud['Class'].empty:
        peak_hour = hour_fraud.loc[hour_fraud['Class'].idxmax(), 'Hour']
        st.info(f"🔥 Peak fraud activity: **{int(peak_hour)}:00 - {int(peak_hour)+1}:00**")

st.caption("💡 Fraud rings often show clustering in time, geography, and merchant networks")
# pages/3_Uplift_Commander.py
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd

st.set_page_config(page_title="Uplift Commander", page_icon="📈", layout="wide")

st.title("📈 Revenue Uplift Commander")

st.markdown("""
Calculate the business impact of reduced false declines.
Adjust parameters to see real-time projections.
""")

# Input parameters
col1, col2, col3 = st.columns(3)

with col1:
    n_txs = st.slider("Monthly Transactions (K)", 100, 10000, 1000, 100) * 1000
    
with col2:
    avg_tx = st.slider("Avg Transaction ($)", 10, 200, 50, 5)
    
with col3:
    baseline_fpr = st.slider("Baseline FPR (%)", 0.0, 10.0, 5.0, 0.1) / 100

# Model performance
model_fpr = 0.01  # 1%
improvement = max(0, baseline_fpr - model_fpr)

legitimate_txs = n_txs * 0.998  # 99.8% legit
false_declines_baseline = legitimate_txs * baseline_fpr
false_declines_model = legitimate_txs * model_fpr
recovered_txs = false_declines_baseline - false_declines_model
monthly_uplift = recovered_txs * avg_tx

st.markdown("## 💰 Revenue Impact")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Recovered Txs/Mo", f"{recovered_txs:,.0f}")
with col2:
    st.metric("Monthly Uplift", f"${monthly_uplift:,.0f}")
with col3:
    st.metric("Annual Uplift", f"${monthly_uplift * 12:,.0f}")
with col4:
    st.metric("FPR Reduction", f"{improvement * 100:.1f}%")

st.markdown("## 📊 Cumulative Revenue Uplift")
months = np.arange(1, 37)
cumulative_uplift = months * monthly_uplift
fig = go.Figure()
fig.add_trace(go.Scatter(
x=months,
y=cumulative_uplift,
mode='lines+markers',
line=dict(color='#10b981', width=4),
marker=dict(size=8, symbol='diamond'),
fill='tozeroy',
fillcolor='rgba(16, 185, 129, 0.2)',
name='Cumulative Uplift'
))
fig.update_layout(
xaxis_title="Months",
yaxis_title="Cumulative Revenue ($)",
paper_bgcolor='rgba(0,0,0,0)',
plot_bgcolor='rgba(30, 41, 59, 0.5)',
font=dict(color='#e2e8f0'),
height=500,
hovermode='x unified'
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("## 📋 3-Year ROI Projection")
implementation_cost = 50000
maintenance_cost_annual = 20000
data = []
for year in range(1, 4):
    annual_revenue = monthly_uplift * 12
    cumulative_revenue = annual_revenue * year
    cumulative_cost = implementation_cost + (maintenance_cost_annual * year)
    net_benefit = cumulative_revenue - cumulative_cost
    roi = (net_benefit / cumulative_cost) * 100 if cumulative_cost > 0 else 0
    data.append({
    'Year': year,
    'Annual Revenue': f"${annual_revenue:,.0f}",
    'Cumulative Revenue': f"${cumulative_revenue:,.0f}",
    'Cumulative Cost': f"${cumulative_cost:,.0f}",
    'Net Benefit': f"${net_benefit:,.0f}",
    'ROI': f"{roi:.0f}%"
})
    st.table(pd.DataFrame(data))
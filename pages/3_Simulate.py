# pages/3_Simulate.py
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.viz import GLASS_THEME

st.set_page_config(page_title="ROI Simulator", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .main { 
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        font-family: 'Inter', sans-serif;
    }
    .roi-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
    }
    .big-number {
        font-size: 3rem;
        font-weight: 800;
        color: #10b981;
        margin: 0;
    }
    .slider-card {
        background: rgba(255, 255, 255, 0.06);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 Revenue Uplift Simulator")
st.markdown("Calculate the business impact of reduced false declines")

# Input sliders
st.markdown("### 🎚️ Simulation Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="slider-card">', unsafe_allow_html=True)
    monthly_txs = st.slider(
        "Monthly Transactions (K)",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        help="Total transaction volume per month"
    ) * 1000
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="slider-card">', unsafe_allow_html=True)
    avg_tx_value = st.slider(
        "Avg Transaction Value ($)",
        min_value=10,
        max_value=500,
        value=75,
        step=5,
        help="Average dollar amount per transaction"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="slider-card">', unsafe_allow_html=True)
    baseline_fpr = st.slider(
        "Baseline False Positive Rate (%)",
        min_value=0.5,
        max_value=10.0,
        value=5.0,
        step=0.1,
        help="Current false decline rate"
    ) / 100
    st.markdown('</div>', unsafe_allow_html=True)

# Model performance
model_fpr = 0.01  # 1% - our model's FPR
fraud_rate = 0.002  # 0.2%

# Calculations
legitimate_txs = monthly_txs * (1 - fraud_rate)
false_declines_baseline = legitimate_txs * baseline_fpr
false_declines_model = legitimate_txs * model_fpr
recovered_txs = max(0, false_declines_baseline - false_declines_model)
monthly_revenue_uplift = recovered_txs * avg_tx_value

fpr_improvement = max(0, baseline_fpr - model_fpr)
uplift_pct = (fpr_improvement / baseline_fpr * 100) if baseline_fpr > 0 else 0

# Display impact
st.markdown("---")
st.markdown("### 💰 Financial Impact")

col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)

with col_metric1:
    st.markdown(f"""
    <div class="roi-card">
        <p style="color: #94a3b8; font-size: 0.875rem; margin: 0;">Recovered Transactions/Mo</p>
        <p class="big-number">{recovered_txs:,.0f}</p>
    </div>
    """, unsafe_allow_html=True)

with col_metric2:
    st.markdown(f"""
    <div class="roi-card">
        <p style="color: #94a3b8; font-size: 0.875rem; margin: 0;">Monthly Revenue Uplift</p>
        <p class="big-number">${monthly_revenue_uplift:,.0f}</p>
    </div>
    """, unsafe_allow_html=True)

with col_metric3:
    st.markdown(f"""
    <div class="roi-card">
        <p style="color: #94a3b8; font-size: 0.875rem; margin: 0;">Annual Revenue Uplift</p>
        <p class="big-number">${monthly_revenue_uplift * 12:,.0f}</p>
    </div>
    """, unsafe_allow_html=True)

with col_metric4:
    st.markdown(f"""
    <div class="roi-card">
        <p style="color: #94a3b8; font-size: 0.875rem; margin: 0;">FPR Reduction</p>
        <p class="big-number">{uplift_pct:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

# Cumulative uplift chart
st.markdown("---")
st.markdown("### 📊 Cumulative Revenue Projection (36 Months)")

months = np.arange(1, 37)
cumulative_revenue = months * monthly_revenue_uplift

fig_cumulative = go.Figure()

fig_cumulative.add_trace(go.Scatter(
    x=months,
    y=cumulative_revenue,
    mode='lines+markers',
    line=dict(color=GLASS_THEME['accent_emerald'], width=4),
    marker=dict(size=8, symbol='diamond', color=GLASS_THEME['accent_emerald']),
    fill='tozeroy',
    fillcolor='rgba(16, 185, 129, 0.2)',
    name='Cumulative Uplift',
    hovertemplate='Month %{x}<br>Revenue: $%{y:,.0f}<extra></extra>'
))

# Breakeven line (if there were implementation costs)
implementation_cost = 50000
fig_cumulative.add_hline(
    y=implementation_cost,
    line_dash="dash",
    line_color=GLASS_THEME['accent_amber'],
    annotation_text=f"Breakeven: ${implementation_cost:,}",
    annotation_position="right"
)

fig_cumulative.update_layout(
    xaxis=dict(
        title="Months Since Deployment",
        gridcolor=GLASS_THEME['grid_color'],
        showgrid=True
    ),
    yaxis=dict(
        title="Cumulative Revenue ($)",
        gridcolor=GLASS_THEME['grid_color'],
        showgrid=True
    ),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor=GLASS_THEME['plot_bg'],
    font=dict(color=GLASS_THEME['text_primary'], family='Inter, sans-serif'),
    height=500,
    hovermode='x unified',
    showlegend=False
)

st.plotly_chart(fig_cumulative, use_container_width=True)

# ROI Table
st.markdown("---")
st.markdown("### 📋 3-Year ROI Breakdown")

maintenance_annual = 20000

roi_data = []
for year in range(1, 4):
    annual_revenue = monthly_revenue_uplift * 12
    cumulative_revenue = annual_revenue * year
    cumulative_cost = implementation_cost + (maintenance_annual * year)
    net_benefit = cumulative_revenue - cumulative_cost
    roi = (net_benefit / cumulative_cost * 100) if cumulative_cost > 0 else 0
    
    roi_data.append({
        'Year': year,
        'Annual Revenue': f'${annual_revenue:,.0f}',
        'Cumulative Revenue': f'${cumulative_revenue:,.0f}',
        'Cumulative Cost': f'${cumulative_cost:,.0f}',
        'Net Benefit': f'${net_benefit:,.0f}',
        'ROI': f'{roi:.0f}%'
    })

df_roi = pd.DataFrame(roi_data)

st.dataframe(
    df_roi.style.set_properties(**{
        'background-color': 'rgba(255, 255, 255, 0.05)',
        'color': '#e2e8f0',
        'border-color': 'rgba(255, 255, 255, 0.1)'
    }),
    use_container_width=True,
    hide_index=True
)

# Cost breakdown
st.markdown("---")
st.markdown("### 💵 Cost vs Revenue Comparison")

col_cost, col_revenue = st.columns(2)

with col_cost:
    # Cost breakdown pie
    costs = {
        'Implementation': implementation_cost,
        'Year 1 Maintenance': maintenance_annual,
        'Year 2 Maintenance': maintenance_annual,
        'Year 3 Maintenance': maintenance_annual
    }
    
    fig_cost = go.Figure(data=[go.Pie(
        labels=list(costs.keys()),
        values=list(costs.values()),
        hole=0.4,
        marker=dict(
            colors=['#ef4444', '#f97316', '#f59e0b', '#eab308'],
            line=dict(color='rgba(255, 255, 255, 0.2)', width=2)
        ),
        textinfo='label+percent',
        textfont=dict(size=11, color='white')
    )])
    
    fig_cost.update_layout(
        title=dict(text="Total Cost Breakdown (3 Years)", font=dict(size=16)),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=GLASS_THEME['text_primary']),
        height=350,
        showlegend=False,
        annotations=[dict(text=f'${sum(costs.values()):,.0f}', x=0.5, y=0.5, 
                          font_size=20, showarrow=False, font_color='#10b981')]
    )
    
    st.plotly_chart(fig_cost, use_container_width=True)

with col_revenue:
    # Revenue by year
    years = ['Year 1', 'Year 2', 'Year 3']
    revenues = [monthly_revenue_uplift * 12] * 3
    
    fig_revenue = go.Figure(data=[go.Bar(
        x=years,
        y=revenues,
        marker=dict(
            color=GLASS_THEME['accent_emerald'],
            line=dict(color='rgba(255, 255, 255, 0.3)', width=2)
        ),
        text=[f'${r:,.0f}' for r in revenues],
        textposition='outside',
        textfont=dict(size=13, color=GLASS_THEME['text_primary'])
    )])
    
    fig_revenue.update_layout(
        title=dict(text="Annual Revenue Uplift", font=dict(size=16)),
        yaxis=dict(title="Revenue ($)", gridcolor=GLASS_THEME['grid_color']),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=GLASS_THEME['plot_bg'],
        font=dict(color=GLASS_THEME['text_primary']),
        height=350,
        showlegend=False
    )
    
    st.plotly_chart(fig_revenue, use_container_width=True)

# Sensitivity analysis
st.markdown("---")
st.markdown("### 🎯 Sensitivity Analysis")

st.info("💡 Explore how changes in key parameters affect ROI")

sensitivity_param = st.selectbox(
    "Vary parameter:",
    ["Transaction Volume", "Average Transaction Value", "Baseline FPR"]
)

if sensitivity_param == "Transaction Volume":
    param_range = np.linspace(monthly_txs * 0.5, monthly_txs * 1.5, 50)
    uplifts = [(p * (1 - fraud_rate) * (baseline_fpr - model_fpr) * avg_tx_value) for p in param_range]
    x_label = "Monthly Transactions"
    x_vals = param_range
elif sensitivity_param == "Average Transaction Value":
    param_range = np.linspace(avg_tx_value * 0.5, avg_tx_value * 1.5, 50)
    uplifts = [recovered_txs * p for p in param_range]
    x_label = "Avg Transaction Value ($)"
    x_vals = param_range
else:  # Baseline FPR
    param_range = np.linspace(0.01, 0.15, 50)
    uplifts = [(legitimate_txs * (p - model_fpr) * avg_tx_value) for p in param_range]
    x_label = "Baseline FPR"
    x_vals = param_range * 100  # Convert to percentage

fig_sensitivity = go.Figure()

fig_sensitivity.add_trace(go.Scatter(
    x=x_vals,
    y=uplifts,
    mode='lines',
    line=dict(color=GLASS_THEME['accent_blue'], width=3),
    fill='tozeroy',
    fillcolor='rgba(59, 130, 246, 0.2)',
    hovertemplate=f'{x_label}: %{{x:.0f}}<br>Monthly Uplift: $%{{y:,.0f}}<extra></extra>'
))

# Current value marker
if sensitivity_param == "Transaction Volume":
    current_val = monthly_txs
elif sensitivity_param == "Average Transaction Value":
    current_val = avg_tx_value
else:
    current_val = baseline_fpr * 100

fig_sensitivity.add_vline(
    x=current_val,
    line_dash="dash",
    line_color=GLASS_THEME['accent_red'],
    annotation_text="Current"
)

fig_sensitivity.update_layout(
    xaxis=dict(title=x_label, gridcolor=GLASS_THEME['grid_color']),
    yaxis=dict(title="Monthly Revenue Uplift ($)", gridcolor=GLASS_THEME['grid_color']),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor=GLASS_THEME['plot_bg'],
    font=dict(color=GLASS_THEME['text_primary']),
    height=400,
    hovermode='x unified'
)

st.plotly_chart(fig_sensitivity, use_container_width=True)

# Footer insights
st.markdown("---")
st.success(f"""
### 🎉 Key Takeaway

By reducing false positive rate from **{baseline_fpr*100:.1f}%** to **{model_fpr*100:.1f}%**, 
FraudShield 2.0 recovers **{recovered_txs:,.0f}** legitimate transactions monthly, 
generating **${monthly_revenue_uplift * 12:,.0f}** in annual revenue uplift.

**Break-even achieved in month {int(np.ceil(implementation_cost / monthly_revenue_uplift))}** 
with **{uplift_pct:.0f}% reduction** in customer friction.
""")

st.caption("💡 Assumptions: 0.2% fraud rate, linear scaling, no operational disruptions")
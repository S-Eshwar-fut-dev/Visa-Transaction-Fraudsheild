# utils/viz.py
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

def create_3d_scatter(df, predictions, x_col='amount_zscore', y_col='contagion_risk', z_col='count_1H'):
    """Create 3D scatter plot of features colored by risk."""
    
    # Ensure columns exist
    if x_col not in df.columns:
        x_col = df.columns[0]
    if y_col not in df.columns:
        y_col = df.columns[1]
    if z_col not in df.columns:
        z_col = df.columns[2]
    
    fig = go.Figure(data=[go.Scatter3d(
        x=df[x_col],
        y=df[y_col],
        z=df[z_col],
        mode='markers',
        marker=dict(
            size=5,
            color=predictions,
            colorscale='RdYlGn_r',
            showscale=True,
            colorbar=dict(title="Risk Score"),
            line=dict(width=0.5, color='rgba(255,255,255,0.3)')
        ),
        text=[f"Risk: {p:.3f}" for p in predictions],
        hovertemplate='<b>%{text}</b><br>' +
                      f'{x_col}: %{{x:.2f}}<br>' +
                      f'{y_col}: %{{y:.2f}}<br>' +
                      f'{z_col}: %{{z:.2f}}<extra></extra>'
    )])
    
    fig.update_layout(
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col,
            bgcolor='rgba(15, 23, 42, 0.5)',
            xaxis=dict(gridcolor='rgba(100, 116, 139, 0.3)'),
            yaxis=dict(gridcolor='rgba(100, 116, 139, 0.3)'),
            zaxis=dict(gridcolor='rgba(100, 116, 139, 0.3)')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        height=600
    )
    
    return fig

def create_risk_gauge(risk_score, threshold=0.4121):
    """Create animated risk gauge."""
    
    color = "#ef4444" if risk_score > threshold else "#10b981"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Fraud Risk", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 1], 'tickcolor': "#64748b"},
            'bar': {'color': color},
            'bgcolor': "rgba(30, 41, 59, 0.3)",
            'borderwidth': 2,
            'bordercolor': "#64748b",
            'steps': [
                {'range': [0, threshold], 'color': 'rgba(16, 185, 129, 0.2)'},
                {'range': [threshold, 1], 'color': 'rgba(239, 68, 68, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "#fbbf24", 'width': 4},
                'thickness': 0.75,
                'value': threshold
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#e2e8f0"},
        height=300
    )
    
    return fig
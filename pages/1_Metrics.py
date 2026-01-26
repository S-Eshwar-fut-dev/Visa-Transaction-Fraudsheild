# pages/1_Metrics.py
import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve, roc_curve, auc, precision_score, recall_score, f1_score, average_precision_score, roc_auc_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import load_model_and_data, predict_batch
from utils.viz import create_pr_gauge, create_feature_importance_bar, GLASS_THEME
from utils.styles import load_css
from utils.components import metric_card, animated_separator

st.set_page_config(page_title="Metrics Dashboard", page_icon="📊", layout="wide")
load_css()

st.title("📊 Performance Metrics Dashboard")
st.markdown("Real-time model evaluation on validation dataset")

# Load resources
model, feature_cols, val_data = load_model_and_data()

if model is None:
    st.error("⚠️ Model not found. Please run training first.")
    st.stop()

# Compute predictions
with st.spinner("Computing predictions..."):
    # Using a subset for faster interaction if needed, but keeping original logic for now
    X_val = val_data[feature_cols].fillna(0)
    y_val = val_data['Class'] if 'Class' in val_data.columns else np.zeros(len(val_data))
    y_pred_proba = predict_batch(model, X_val)
    
    threshold = 0.4121  # From training

# Metrics computation
y_pred = (y_pred_proba >= threshold).astype(int)

pr_auc = average_precision_score(y_val, y_pred_proba) if y_val.sum() > 0 else 0
roc_auc = roc_auc_score(y_val, y_pred_proba) if y_val.sum() > 0 else 0
precision = precision_score(y_val, y_pred) if y_val.sum() > 0 else 0
recall = recall_score(y_val, y_pred) if y_val.sum() > 0 else 0
f1 = f1_score(y_val, y_pred) if y_val.sum() > 0 else 0

# Top metrics in glass cards
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    metric_card("PR-AUC", f"{pr_auc:.4f}", "Target: >0.80")
with col2:
    metric_card("ROC-AUC", f"{roc_auc:.4f}", "Target: >0.95")
with col3:
    metric_card("Precision", f"{precision:.3f}", "High Trust")
with col4:
    metric_card("Recall", f"{recall:.3f}", "Broad Coverage")
with col5:
    metric_card("F1 Score", f"{f1:.3f}", "Balanced")

animated_separator()

# PR-AUC Gauge + Feature Importance
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("### 🎯 PR-AUC Performance")
    fig_gauge = create_pr_gauge(pr_auc=pr_auc, target=0.80)
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Confusion matrix
    st.markdown("### 📊 Confusion Matrix")
    cm = confusion_matrix(y_val, y_pred)
    
    fig_cm = go.Figure(data=go.Heatmap(
        z=[[cm[1,1], cm[1,0]], [cm[0,1], cm[0,0]]],
        x=['Predicted Fraud', 'Predicted Legit'],
        y=['Actual Fraud', 'Actual Legit'],
        text=[[cm[1,1], cm[1,0]], [cm[0,1], cm[0,0]]],
        texttemplate='%{text}',
        colorscale='Teal',
        showscale=False
    ))
    
    fig_cm.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=GLASS_THEME['plot_bg'],
        font=dict(color=GLASS_THEME['text_primary']),
        height=350,
        margin=dict(l=50, r=20, t=30, b=50)
    )
    
    st.plotly_chart(fig_cm, use_container_width=True)

with col_right:
    st.markdown("### 🏆 Feature Importance")
    
    # Extract feature importance
    try:
        importance_dict = model.get_score(importance_type='gain')
        importance_df = pd.DataFrame({
            'feature': list(importance_dict.keys()),
            'importance': list(importance_dict.values())
        }).sort_values('importance', ascending=False)
        
        fig_importance = create_feature_importance_bar(importance_df, top_n=15)
        st.plotly_chart(fig_importance, use_container_width=True)
    except:
        st.warning("Feature importance not available")

animated_separator()

# PR and ROC Curves
col_pr, col_roc = st.columns(2)

with col_pr:
    st.markdown("### 📈 Precision-Recall Curve")
    
    if y_val.sum() > 0:
        precision_vals, recall_vals, _ = precision_recall_curve(y_val, y_pred_proba)
        
        fig_pr = go.Figure()
        
        fig_pr.add_trace(go.Scatter(
            x=recall_vals,
            y=precision_vals,
            mode='lines',
            fill='tozeroy',
            fillcolor='rgba(16, 185, 129, 0.2)',
            line=dict(color=GLASS_THEME['accent_emerald'], width=3),
            name=f'PR-AUC = {pr_auc:.3f}',
            hovertemplate='Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>'
        ))
        
        # Add baseline
        baseline = y_val.sum() / len(y_val)
        fig_pr.add_hline(
            y=baseline,
            line_dash="dash",
            line_color=GLASS_THEME['text_muted'],
            annotation_text=f"Random (baseline={baseline:.3f})"
        )
        
        fig_pr.update_layout(
            xaxis_title="Recall",
            yaxis_title="Precision",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor=GLASS_THEME['plot_bg'],
            font=dict(color=GLASS_THEME['text_primary']),
            height=400,
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                bgcolor='rgba(30, 41, 59, 0.8)',
                bordercolor=GLASS_THEME['card_border'],
                borderwidth=1
            )
        )
        
        st.plotly_chart(fig_pr, use_container_width=True)
    else:
        st.info("No fraud cases in validation set")

with col_roc:
    st.markdown("### 📉 ROC Curve")
    
    if y_val.sum() > 0:
        fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
        
        fig_roc = go.Figure()
        
        fig_roc.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            fill='tozeroy',
            fillcolor='rgba(59, 130, 246, 0.2)',
            line=dict(color=GLASS_THEME['accent_blue'], width=3),
            name=f'ROC-AUC = {roc_auc:.3f}',
            hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
        ))
        
        # Diagonal reference
        fig_roc.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            line=dict(color=GLASS_THEME['text_muted'], dash='dash'),
            name='Random',
            showlegend=False
        ))
        
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor=GLASS_THEME['plot_bg'],
            font=dict(color=GLASS_THEME['text_primary']),
            height=400,
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                bgcolor='rgba(30, 41, 59, 0.8)',
                bordercolor=GLASS_THEME['card_border'],
                borderwidth=1
            )
        )
        
        st.plotly_chart(fig_roc, use_container_width=True)
    else:
        st.info("No fraud cases in validation set")

# Threshold analysis
animated_separator()
st.markdown("### 🎚️ Threshold Analysis")

if y_val.sum() > 0:
    threshold_range = np.linspace(0, 1, 100)
    precisions_at_thresh = []
    recalls_at_thresh = []
    f1s_at_thresh = []
    
    for thresh in threshold_range:
        y_pred_thresh = (y_pred_proba >= thresh).astype(int)
        prec = precision_score(y_val, y_pred_thresh, zero_division=0)
        rec = recall_score(y_val, y_pred_thresh, zero_division=0)
        f1_score_val = f1_score(y_val, y_pred_thresh, zero_division=0)
        
        precisions_at_thresh.append(prec)
        recalls_at_thresh.append(rec)
        f1s_at_thresh.append(f1_score_val)
    
    fig_thresh = go.Figure()
    
    fig_thresh.add_trace(go.Scatter(
        x=threshold_range,
        y=precisions_at_thresh,
        mode='lines',
        line=dict(color=GLASS_THEME['accent_emerald'], width=2),
        name='Precision'
    ))
    
    fig_thresh.add_trace(go.Scatter(
        x=threshold_range,
        y=recalls_at_thresh,
        mode='lines',
        line=dict(color=GLASS_THEME['accent_blue'], width=2),
        name='Recall'
    ))
    
    fig_thresh.add_trace(go.Scatter(
        x=threshold_range,
        y=f1s_at_thresh,
        mode='lines',
        line=dict(color=GLASS_THEME['accent_amber'], width=2),
        name='F1 Score'
    ))
    
    # Current threshold
    fig_thresh.add_vline(
        x=threshold,
        line_dash="dash",
        line_color=GLASS_THEME['accent_red'],
        annotation_text=f"Current: {threshold:.4f}"
    )
    
    fig_thresh.update_layout(
        xaxis_title="Decision Threshold",
        yaxis_title="Score",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=GLASS_THEME['plot_bg'],
        font=dict(color=GLASS_THEME['text_primary']),
        height=400,
        hovermode='x unified',
        legend=dict(
            bgcolor='rgba(30, 41, 59, 0.8)',
            bordercolor=GLASS_THEME['card_border'],
            borderwidth=1
        )
    )
    
    st.plotly_chart(fig_thresh, use_container_width=True)
else:
    st.info("Threshold analysis requires fraud cases")

st.caption("💡 Tip: Adjust threshold in production based on business risk tolerance")
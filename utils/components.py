import streamlit as st
import plotly.graph_objects as go
from utils.styles import GLASS_THEME

def metric_card(title, value, delta=None, delta_color="normal"):
    """
    Renders a glassmorphic metric card.
    
    Args:
        title (str): Metric label
        value (str): Main value to display
        delta (str): Change text (optional)
        delta_color (str): 'normal' (green/red based on +/-) or 'inverse' or specific color hex
    """
    
    color_style = ""
    arrow = ""
    
    if delta:
        if delta_color == "normal":
            is_positive = "+" in delta or "↑" in delta
            color = "var(--accent-emerald)" if is_positive else "var(--accent-rose)"
            arrow = "↑" if is_positive else "↓"
        else:
            color = delta_color
            
        color_style = f"color: {color};"
        if arrow not in delta:
            delta = f"{arrow} {delta}"
            
    html = f"""
    <div class="glass-card animate-enter">
        <div class="metric-container">
            <div class="metric-label">{title}</div>
            <div class="metric-value">{value}</div>
            {f'<div class="metric-delta"><span style="{color_style}">{delta}</span></div>' if delta else ''}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def animated_separator():
    st.markdown("""
    <div style="height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent); margin: 3rem 0;"></div>
    """, unsafe_allow_html=True)

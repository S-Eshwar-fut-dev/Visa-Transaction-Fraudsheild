import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import networkx as nx
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.data_loader import load_model_and_data

st.set_page_config(page_title="Ring Hunter", page_icon="🕸️", layout="wide")

st.title("🕸️ Fraud Ring Hunter")

try:
    model, feature_cols, val_data = load_model_and_data()
    
    # Filter high-risk transactions
    if 'contagion_risk' in val_data.columns:
        ring_threshold = st.sidebar.slider("Contagion Threshold", 0.0, 0.5, 0.1, 0.01)
        ring_data = val_data[val_data['contagion_risk'] > ring_threshold]
    else:
        ring_data = val_data[val_data['Class'] == 1]
    
    st.sidebar.metric("Potential Ring Members", len(ring_data))
    
    # Geo map
    st.markdown("## 🗺️ Geographic Distribution")
    
    if 'V4' in ring_data.columns and 'V11' in ring_data.columns:
        # Create map
        m = folium.Map(location=[37.7749, -95.4194], zoom_start=4)
        
        for _, row in ring_data.head(50).iterrows():
            folium.CircleMarker(
                location=[row['V4'], row['V11']],
                radius=5,
                popup=f"Amount: ${row['Amount']:.2f}",
                color='red' if row['Class'] == 1 else 'orange',
                fill=True,
                fillColor='red' if row['Class'] == 1 else 'orange'
            ).add_to(m)
        
        st_folium(m, width=1200, height=500)
    else:
        st.info("Geographic data not available")
    
    # Network graph
    st.markdown("## 🕸️ Transaction Network")
    
    # Create synthetic network
    G = nx.Graph()
    
    # Sample users and merchants
    sample_size = min(20, len(ring_data))
    sample = ring_data.sample(sample_size)
    
    for _, row in sample.iterrows():
        user = f"U{row['user_id']}"
        merchant = f"M{row['merchant_id']}"
        G.add_edge(user, merchant, weight=row['Amount'])
    
    # Layout
    pos = nx.spring_layout(G, seed=42)
    
    # Create plotly network
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='rgba(100, 116, 139, 0.5)'),
        hoverinfo='none',
        mode='lines')
    
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_color.append('#ef4444' if node.startswith('U') else '#3b82f6')
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        marker=dict(
            size=20,
            color=node_color,
            line=dict(width=2, color='white')),
        hoverinfo='text')
    
    fig = go.Figure(data=[edge_trace, node_trace])
    
    fig.update_layout(
        title="User-Merchant Network (Red=Users, Blue=Merchants)",
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30, 41, 59, 0.5)',
        font=dict(color='#e2e8f0'),
        height=600,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Error: {str(e)}")

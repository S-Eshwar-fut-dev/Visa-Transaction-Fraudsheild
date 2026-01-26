# utils/viz.py
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

from utils.styles import GLASS_THEME

def create_3d_scatter(df, predictions, x_col='amount_zscore', y_col='contagion_risk', z_col='count_1h'):
    """Create immersive 3D feature space visualization."""
    
    # Validate columns exist
    x_col = x_col if x_col in df.columns else df.select_dtypes(include=[np.number]).columns[0]
    y_col = y_col if y_col in df.columns else df.select_dtypes(include=[np.number]).columns[1]
    z_col = z_col if z_col in df.columns else df.select_dtypes(include=[np.number]).columns[2]
    
    # Create fraud mask
    fraud_mask = df['Class'] == 1 if 'Class' in df.columns else predictions > 0.5
    
    fig = go.Figure()
    
    # Legitimate transactions (green-blue gradient)
    legit_idx = ~fraud_mask
    fig.add_trace(go.Scatter3d(
        x=df.loc[legit_idx, x_col],
        y=df.loc[legit_idx, y_col],
        z=df.loc[legit_idx, z_col],
        mode='markers',
        name='Legitimate',
        marker=dict(
            size=4,
            color=predictions[legit_idx],
            colorscale='Teal',
            opacity=0.6,
            line=dict(width=0.3, color='rgba(255,255,255,0.3)'),
            showscale=False
        ),
        hovertemplate='<b>Legit TX</b><br>Risk: %{marker.color:.3f}<br>' +
                      f'{x_col}: %{{x:.2f}}<br>{y_col}: %{{y:.2f}}<br>{z_col}: %{{z:.2f}}<extra></extra>'
    ))
    
    # Fraud transactions (red glow)
    fraud_idx = fraud_mask
    if fraud_idx.sum() > 0:
        fig.add_trace(go.Scatter3d(
            x=df.loc[fraud_idx, x_col],
            y=df.loc[fraud_idx, y_col],
            z=df.loc[fraud_idx, z_col],
            mode='markers',
            name='Fraud',
            marker=dict(
                size=8,
                color=GLASS_THEME['accent_red'],
                symbol='diamond',
                opacity=0.9,
                line=dict(width=2, color='#fca5a5')
            ),
            hovertemplate='<b>FRAUD</b><br>Risk: %{marker.color:.3f}<br>' +
                          f'{x_col}: %{{x:.2f}}<br>{y_col}: %{{y:.2f}}<br>{z_col}: %{{z:.2f}}<extra></extra>'
        ))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title=x_col.replace('_', ' ').title(),
                backgroundcolor=GLASS_THEME['plot_bg'],
                gridcolor=GLASS_THEME['grid_color'],
                showbackground=True
            ),
            yaxis=dict(
                title=y_col.replace('_', ' ').title(),
                backgroundcolor=GLASS_THEME['plot_bg'],
                gridcolor=GLASS_THEME['grid_color'],
                showbackground=True
            ),
            zaxis=dict(
                title=z_col.replace('_', ' ').title(),
                backgroundcolor=GLASS_THEME['plot_bg'],
                gridcolor=GLASS_THEME['grid_color'],
                showbackground=True
            ),
            bgcolor=GLASS_THEME['paper_bg']
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=GLASS_THEME['text_primary'], family='Inter, sans-serif'),
        height=650,
        showlegend=True,
        legend=dict(
            bgcolor='rgba(30, 41, 59, 0.8)',
            bordercolor=GLASS_THEME['card_border'],
            borderwidth=1,
            font=dict(size=11)
        ),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return fig

def create_pr_gauge(pr_auc=0.839, target=0.80):
    """Animated PR-AUC gauge with glassmorphic styling."""
    
    color = GLASS_THEME['accent_emerald'] if pr_auc >= target else GLASS_THEME['accent_amber']
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pr_auc,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "PR-AUC Score", 'font': {'size': 22, 'color': GLASS_THEME['text_primary']}},
        delta={'reference': target, 'increasing': {'color': GLASS_THEME['accent_emerald']}},
        number={'suffix': '', 'font': {'size': 48, 'color': color}},
        gauge={
            'axis': {'range': [0, 1], 'tickwidth': 2, 'tickcolor': GLASS_THEME['text_muted']},
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': 'rgba(30, 41, 59, 0.4)',
            'borderwidth': 3,
            'bordercolor': GLASS_THEME['card_border'],
            'steps': [
                {'range': [0, 0.6], 'color': 'rgba(239, 68, 68, 0.15)'},
                {'range': [0.6, 0.8], 'color': 'rgba(251, 191, 36, 0.15)'},
                {'range': [0.8, 1], 'color': 'rgba(16, 185, 129, 0.15)'}
            ],
            'threshold': {
                'line': {'color': GLASS_THEME['accent_blue'], 'width': 5},
                'thickness': 0.85,
                'value': pr_auc
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': GLASS_THEME['text_primary'], 'family': 'Inter, sans-serif'},
        height=340,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def create_feature_importance_bar(importance_df, top_n=15):
    """Horizontal bar chart for feature importance."""
    
    top_features = importance_df.head(top_n).copy()
    
    # Normalize for color gradient
    top_features['norm'] = (top_features['importance'] - top_features['importance'].min()) / \
                            (top_features['importance'].max() - top_features['importance'].min())
    
    colors = [f'rgba(16, 185, 129, {0.4 + 0.6*val})' for val in top_features['norm']]
    
    fig = go.Figure(go.Bar(
        y=top_features['feature'],
        x=top_features['importance'],
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color=GLASS_THEME['accent_emerald'], width=1.5)
        ),
        text=top_features['importance'].round(1),
        textposition='outside',
        textfont=dict(color=GLASS_THEME['text_primary'], size=11),
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.1f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text="🏆 Top Features by Gain",
            font=dict(size=18, color=GLASS_THEME['text_primary'])
        ),
        xaxis=dict(
            title="Importance Score",
            gridcolor=GLASS_THEME['grid_color'],
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=11)
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=GLASS_THEME['plot_bg'],
        font=dict(color=GLASS_THEME['text_primary'], family='Inter, sans-serif'),
        height=500,
        margin=dict(l=150, r=50, t=60, b=50),
        hovermode='y unified'
    )
    
    return fig

def create_shap_waterfall(shap_values, feature_names, base_value=0.002, top_k=10):
    """SHAP waterfall chart (alternative to force plot)."""
    
    # Get top contributors by absolute value
    abs_values = np.abs(shap_values)
    top_idx = np.argsort(abs_values)[-top_k:][::-1]
    
    features = [feature_names[i] for i in top_idx]
    values = [shap_values[i] for i in top_idx]
    
    # Cumulative values for waterfall
    cumulative = np.cumsum([base_value] + values)
    
    colors = [GLASS_THEME['accent_red'] if v > 0 else GLASS_THEME['accent_emerald'] for v in values]
    
    fig = go.Figure()
    
    # Base value
    fig.add_trace(go.Bar(
        x=[base_value],
        y=['Base Rate'],
        orientation='h',
        marker=dict(color=GLASS_THEME['text_muted']),
        text=f"{base_value:.4f}",
        textposition='outside',
        name='Base'
    ))
    
    # Contributions
    for i, (feat, val, color) in enumerate(zip(features, values, colors)):
        fig.add_trace(go.Bar(
            x=[abs(val)],
            y=[feat],
            orientation='h',
            marker=dict(color=color),
            text=f"{val:+.4f}",
            textposition='outside',
            name=feat,
            showlegend=False
        ))
    
    fig.update_layout(
        title=dict(
            text="🔬 SHAP Value Contributions",
            font=dict(size=18, color=GLASS_THEME['text_primary'])
        ),
        xaxis=dict(
            title="SHAP Value (log-odds)",
            gridcolor=GLASS_THEME['grid_color']
        ),
        yaxis=dict(title=""),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=GLASS_THEME['plot_bg'],
        font=dict(color=GLASS_THEME['text_primary'], family='Inter, sans-serif'),
        height=450,
        margin=dict(l=180, r=100, t=60, b=50),
        barmode='relative'
    )
    
    return fig

def create_network_graph(user_merchant_edges, fraud_nodes=None):
    """Network visualization for fraud rings."""
    
    import networkx as nx
    
    G = nx.Graph()
    for user, merchant, weight in user_merchant_edges:
        G.add_edge(user, merchant, weight=weight)
    
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    # Extract edges
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='rgba(100, 116, 139, 0.4)'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Extract nodes
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        is_user = node.startswith('U') or node.startswith('u')
        is_fraud = fraud_nodes and node in fraud_nodes
        
        node_text.append(node)
        node_color.append(GLASS_THEME['accent_red'] if is_fraud else 
                         (GLASS_THEME['accent_blue'] if is_user else GLASS_THEME['accent_emerald']))
        node_size.append(20 if is_fraud else 15)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        textfont=dict(size=9, color=GLASS_THEME['text_primary']),
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=2, color='rgba(255,255,255,0.6)')
        ),
        hoverinfo='text',
        hovertext=node_text
    )
    
    fig = go.Figure(data=[edge_trace, node_trace])
    
    fig.update_layout(
        title=dict(
            text="🕸️ User-Merchant Network",
            font=dict(size=18, color=GLASS_THEME['text_primary'])
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=GLASS_THEME['plot_bg'],
        font=dict(color=GLASS_THEME['text_primary']),
        height=600,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig
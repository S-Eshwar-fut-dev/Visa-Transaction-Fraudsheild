import streamlit as st

def load_css():
    """Injects global CSS for glassmorphism, animations, and typography."""
    st.markdown("""
    <style>
        /* Import Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap');
        
        :root {
            --bg-dark: #020617;
            --bg-card: rgba(255, 255, 255, 0.03);
            --border-glass: rgba(255, 255, 255, 0.08);
            --text-primary: #e2e8f0;
            --text-secondary: #94a3b8;
            --accent-emerald: #10b981;
            --accent-blue: #3b82f6;
            --accent-purple: #8b5cf6;
            --accent-rose: #f43f5e;
            --glow-primary: rgba(59, 130, 246, 0.5);
        }

        /* Base Reset */
        .stApp {
            background: radial-gradient(circle at 50% 0%, #1e1b4b 0%, #020617 60%);
            font-family: 'Inter', sans-serif;
            color: var(--text-primary);
        }

        /* Aurora Background Animation */
        .main::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: 
                radial-gradient(circle at 15% 50%, rgba(59, 130, 246, 0.08), transparent 25%),
                radial-gradient(circle at 85% 30%, rgba(16, 185, 129, 0.08), transparent 25%);
            pointer-events: none;
            z-index: -1;
            animation: pulse-glow 10s ease-in-out infinite alternate;
        }

        @keyframes pulse-glow {
            0% { opacity: 0.5; transform: scale(1); }
            100% { opacity: 0.8; transform: scale(1.1); }
        }

        /* Glassmorphic Card */
        .glass-card {
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--border-glass);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
        }

        .glass-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
            border-color: rgba(255, 255, 255, 0.2);
        }

        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            font-weight: 700;
            letter-spacing: -0.025em;
            color: white;
        }

        .gradient-text {
            background: linear-gradient(135deg, #60a5fa 0%, #34d399 50%, #a78bfa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        /* Custom Metrics */
        .metric-container {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .metric-label {
            font-size: 0.875rem;
            color: var(--text-secondary);
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .metric-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 2rem;
            font-weight: 700;
            color: white;
            text-shadow: 0 0 20px rgba(255, 255, 255, 0.1);
        }

        .metric-delta {
            font-size: 0.875rem;
            display: flex;
            align-items: center;
            gap: 0.25rem;
        }
        
        .delta-pos { color: var(--accent-emerald); }
        .delta-neg { color: var(--accent-rose); }

        /* Animations */
        @keyframes slideUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .animate-enter {
            animation: slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #0f172a;
        }
        ::-webkit-scrollbar-thumb {
            background: #334155;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #475569;
        }

        /* Streamlit Overrides */
        [data-testid="stSidebar"] {
            background-color: rgba(2, 6, 23, 0.95);
            border-right: 1px solid var(--border-glass);
        }
        
        .stButton button {
            background: linear-gradient(to right, #2563eb, #4f46e5);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s;
            width: 100%;
        }

        .stButton button:hover {
            box-shadow: 0 0 15px rgba(59, 130, 246, 0.5);
            transform: scale(1.02);
        }
        
        /* Remove Streamlit branding */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        
    </style>
    """, unsafe_allow_html=True)

GLASS_THEME = {
    'bg_gradient': 'linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%)',
    'card_bg': 'rgba(255, 255, 255, 0.08)',
    'card_border': 'rgba(255, 255, 255, 0.15)',
    'text_primary': '#e2e8f0',
    'text_secondary': '#94a3b8',
    'text_muted': '#94a3b8',
    'accent_emerald': '#10b981',
    'accent_blue': '#3b82f6',
    'accent_red': '#ef4444',
    'accent_amber': '#fbbf24',
    'accent_rose': '#f43f5e',
    'accent_purple': '#8b5cf6',
    'grid_color': 'rgba(100, 116, 139, 0.2)',
    'paper_bg': 'rgba(15, 23, 42, 0.3)',
    'plot_bg': 'rgba(30, 41, 59, 0.4)'
}

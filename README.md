# 🛡️ Sigma FraudShield 2.0
> **AI-Powered Real-Time Fraud Detection Command Center**

![UI Preview](https://img.shields.io/badge/Status-Active-success?style=for-the-badge) ![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge) ![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-ff4b4b?style=for-the-badge)

Sigma FraudShield 2.0 is a next-generation fraud detection platform designed for fintech and banking environments. It combines state-of-the-art machine learning (XGBoost) with a futuristic, glassmorphic UI to provide real-time insights, explainable AI (XAI), and deep forensic capabilities.

---

## 🚀 Key Features

### 1. **Immersive Command Center (`app.py`)**
*   **Glassmorphic Design System**: sleek, dark-mode UI with aurora gradients, frosted glass cards, and smooth CSS animations.
*   **Real-Time Monitoring**: Live ticker for incoming transactions and system health indicators.
*   **Key Metrics**: 3D tilting cards displaying critical KPIs like Precision, Recall, and Blocked Fraud Value.

### 2. **Advanced Metrics Dashboard (`pages/1_Metrics.py`)**
*   **Performance Tracking**: Live evaluation on validation data using PR-AUC and ROC-AUC scores.
*   **Interactive Visuals**: Synchronized Precision-Recall vs. Threshold charts and animated confusion matrices.
*   **Feature Importance**: Dynamic bar charts highlighting the top drivers of fraud.

### 3. **Explainability Lab (`pages/2_Explain.py`)**
*   **Transaction "Ticket" View**: Detailed breakdown of individual transactions with risk scores and metadata.
*   **SHAP Waterfall Charts**: Interactive explanation of *why* a specific transaction was flagged.
*   **Visa-Style Reason Codes**: Automated mapping of risk factors to standardized reason codes (e.g., `R02: Fraud Ring Association`).

### 4. **ROI Simulator (`pages/3_Simulate.py`)**
*   **Business Impact Analysis**: Calculate monthly and annual revenue uplift from reduced false declines.
*   **Sensitivity Analysis**: Interactive 3D and 2D charts exploring how changes in transaction volume or model performance affect the bottom line.
*   **Financial Projections**: Cumulative revenue growth forecasts over a 36-month horizon.

### 5. **Fraud Ring Hunter (`pages/4_Rings.py`)**
*   **3D Feature Space**: Explore high-dimensional transaction data in an interactive 3D scatter plot.
*   **Network Graphs**: Visualize connections between Users and Merchants to detect organized fraud rings.
*   **Geospatial Analysis**: Heatmaps of fraudulent activity (requires location data).

---

## 🛠️ Technology Stack

*   **Frontend**: [Streamlit](https://streamlit.io/), Custom CSS3, HTML5
*   **Visualization**: [Plotly](https://plotly.com/), Altair
*   **Machine Learning**: XGBoost, Scikit-learn, Shapley Additive Explanations (SHAP)
*   **Data Processing**: Pandas, NumPy
*   **Design**: Custom "Glassmorphic" Design System (`utils/styles.py`)

---

## 📦 Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/S-Eshwar-fut-dev/Visa-Transaction-Fraudsheild.git
    cd Visa-Transaction-Fraudsheild
    ```

2.  **Create a virtual environment**
    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate

    # Mac/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚦 Usage

1.  **Run the application**
    ```bash
    streamlit run app.py
    ```

2.  **Navigation**
    *   Use the sidebar to navigate between the Command Center, Metrics, Explainability, Simulator, and Ring Hunter pages.
    *   Adjust parameters in the sidebar of individual pages to filter data or simulate scenarios.

---

## 📂 Project Structure

```text
Visa-Transaction-Fraudsheild/
├── app.py                  # Main Entry Point (Command Center)
├── pages/
│   ├── 1_Metrics.py        # Model Performance Dashboard
│   ├── 2_Explain.py        # SHAP Explainability Lab
│   ├── 3_Simulate.py       # ROI & Business Value Simulator
│   └── 4_Rings.py          # Fraud Ring Detection & Graph Analysis
├── utils/
│   ├── components.py       # Reusable UI Components (Cards, Metrics)
│   ├── data_loader.py      # Data Ingestion & Model Loading
│   ├── styles.py           # Global CSS & Design System
│   └── viz.py              # Plotly Visualization Functions
├── data/                   # Dataset Directory (Git-ignored)
├── models/                 # Saved Model Artifacts
├── requirements.txt        # Python Dependencies
└── README.md               # Project Documentation
```

---

## 👥 Contributors

*   **Eshwar S** - *Lead Developer & UI/UX Architect*

---

> **Note**: This project uses a synthetic dataset for demonstration purposes. The "AI Co-Pilot" features in the Explainability Lab are currently simulated visuals.

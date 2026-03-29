import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import os
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
}

.main { background-color: #0f1117; }

.metric-card {
    background: linear-gradient(135deg, #1a1d2e 0%, #16213e 100%);
    border: 1px solid #2d3561;
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-3px); }
.metric-card .label {
    font-size: 12px;
    color: #8892b0;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 500;
    margin-bottom: 8px;
}
.metric-card .value {
    font-size: 32px;
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    color: #64ffda;
}
.metric-card .sub {
    font-size: 11px;
    color: #556;
    margin-top: 4px;
}

.winner-badge {
    background: linear-gradient(90deg, #64ffda22, #64ffda11);
    border: 1px solid #64ffda55;
    border-radius: 8px;
    padding: 12px 20px;
    margin: 8px 0;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    color: #64ffda;
    font-size: 14px;
}

.risk-high {
    background: linear-gradient(135deg, #ff6b6b22, #ff6b6b11);
    border: 1px solid #ff6b6b66;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.risk-low {
    background: linear-gradient(135deg, #64ffda22, #64ffda11);
    border: 1px solid #64ffda66;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}

.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 800;
    color: #ccd6f6;
    border-left: 4px solid #64ffda;
    padding-left: 14px;
    margin: 28px 0 18px 0;
}

.rule-box {
    background: #1a1d2e;
    border: 1px solid #2d3561;
    border-radius: 10px;
    padding: 16px;
    margin: 8px 0;
    font-size: 13px;
    color: #8892b0;
    font-family: 'DM Mono', monospace;
}

div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid #21262d;
}

.stSelectbox > div > div {
    background-color: #1a1d2e !important;
    border-color: #2d3561 !important;
    color: #ccd6f6 !important;
}

div[data-testid="metric-container"] {
    background: #1a1d2e;
    border: 1px solid #2d3561;
    border-radius: 12px;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  LOAD MODELS & DATA
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        'Logistic Regression': 'models/logistic_regression.pkl',
        'Decision Tree':       'models/decision_tree.pkl',
        'Random Forest':       'models/random_forest.pkl',
        'XGBoost':             'models/xgboost.pkl',
    }
    for name, path in model_files.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
    scaler = joblib.load('models/scaler.pkl') if os.path.exists('models/scaler.pkl') else None
    return models, scaler

@st.cache_data
def load_comparison_data():
    path = 'Results/model_comparison.csv'
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    # Fallback — your actual results
    data = {
        'Model':     ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost', 'Rule-Based'],
        'Accuracy':  [0.7393, 0.7037, 0.7607, 0.7229, 0.7393],
        'Precision': [0.5044, 0.4632, 0.5337, 0.4858, 0.6087],
        'Recall':    [0.7655, 0.7628, 0.7466, 0.8302, 0.0377],
        'F1 Score':  [0.6081, 0.5764, 0.6225, 0.6129, 0.0711],
        'ROC-AUC':   [0.8320, 0.8080, 0.8370, 0.8297, None],
    }
    return pd.DataFrame(data).set_index('Model')

models, scaler = load_models()
df_compare = load_comparison_data()


# ─────────────────────────────────────────────────────────────
#  RULE-BASED FUNCTION
# ─────────────────────────────────────────────────────────────
def rule_based_predict(tenure, income, purchases, is_senior, membership):
    month_to_month = 1 if membership == 'Month-to-month' else 0
    two_year       = 1 if membership == 'Two year' else 0
    charges_per_service = income / (purchases + 1)

    if month_to_month and tenure < 12:
        return 1, "Month-to-month contract + tenure < 12 months"
    if income > 70 and purchases <= 2:
        return 1, "High charges but very few services"
    if is_senior and month_to_month:
        return 1, "Senior citizen on short-term contract"
    if month_to_month and tenure < 24:
        return 1, "Month-to-month contract + tenure < 24 months"
    if charges_per_service > 25 and not two_year:
        return 1, "High charge-per-service ratio without long-term commitment"
    return 0, "No high-risk patterns detected"


# ─────────────────────────────────────────────────────────────
#  SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px 0;'>
        <div style='font-family: Syne; font-size: 26px; font-weight: 800; color: #64ffda;'>📉 ChurnScope</div>
        <div style='font-size: 11px; color: #556; margin-top: 4px; letter-spacing: 2px;'>ML DASHBOARD</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Overview", "📊 Model Comparison", "🔮 Live Prediction", "📋 Rule-Based Logic"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size: 11px; color: #445; padding: 10px 0; line-height: 1.8;'>
    <b style='color:#667'>Dataset</b><br>IBM Telco Churn<br>
    <b style='color:#667'>Customers</b><br>7,043 records<br>
    <b style='color:#667'>Churn Rate</b><br>26.5%<br>
    <b style='color:#667'>Best Model</b><br>XGBoost (Recall: 83%)
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  PAGE 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.markdown("""
    <h1 style='color:#ccd6f6; margin-bottom:4px;'>Customer Churn Prediction</h1>
    <p style='color:#8892b0; font-size:15px; margin-bottom:32px;'>
    AI & Machine Learning Internship Assessment — Model Performance Dashboard
    </p>
    """, unsafe_allow_html=True)

    # Top metric cards
    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        ("Total Customers", "7,043", "IBM Telco Dataset"),
        ("Churned", "1,869", "26.5% churn rate"),
        ("Models Trained", "4 + 1", "ML + Rule-Based"),
        ("Best Recall", "83.0%", "XGBoost"),
        ("Best AUC", "0.837", "Random Forest"),
    ]
    for col, (label, value, sub) in zip([c1,c2,c3,c4,c5], cards):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='label'>{label}</div>
                <div class='value'>{value}</div>
                <div class='sub'>{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("<div class='section-header'>Model Performance at a Glance</div>", unsafe_allow_html=True)

        plot_df = df_compare.copy()
        if 'ROC-AUC' in plot_df.columns:
            plot_df = plot_df.drop(columns=['ROC-AUC'], errors='ignore')
        plot_df = plot_df[['Accuracy','Precision','Recall','F1 Score']]

        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor('#0f1117')
        ax.set_facecolor('#1a1d2e')

        x = np.arange(len(plot_df.index))
        width = 0.2
        colors = ['#64ffda', '#ff6b9d', '#ffd166', '#a78bfa']
        labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

        for i, (col, color, label) in enumerate(zip(plot_df.columns, colors, labels)):
            bars = ax.bar(x + i*width, plot_df[col], width,
                         label=label, color=color, alpha=0.85, zorder=3)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.01,
                        f'{bar.get_height():.2f}',
                        ha='center', va='bottom',
                        fontsize=7, color='#8892b0')

        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(plot_df.index, rotation=12, color='#ccd6f6', fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel('Score', color='#8892b0', fontsize=10)
        ax.tick_params(colors='#8892b0')
        ax.legend(fontsize=9, labelcolor='#ccd6f6',
                  facecolor='#1a1d2e', edgecolor='#2d3561')
        ax.grid(axis='y', color='#2d3561', alpha=0.5, zorder=0)
        ax.spines[['top','right','left','bottom']].set_color('#2d3561')
        ax.set_title('All Models — All Metrics', color='#ccd6f6',
                     fontsize=12, fontweight='bold', pad=12)

        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("<div class='section-header'>Winner & Key Finding</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='winner-badge'>🏆 XGBoost — Best Recall (83.02%)</div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style='color:#8892b0; font-size:13px; line-height:1.8; margin-top:12px;'>
        XGBoost caught <b style='color:#64ffda'>83%</b> of all actual churners —
        the highest of all models. In churn prediction, <b style='color:#ffd166'>Recall
        matters more than Accuracy</b> because missing a churning customer
        costs far more than a false alarm.<br><br>
        Random Forest had the best overall AUC at <b style='color:#64ffda'>0.837</b>,
        making it the most balanced model across all metrics.<br><br>
        The Rule-Based system had strong Precision (<b style='color:#ff6b9d'>0.608</b>)
        but almost zero Recall (<b style='color:#ff6b9d'>3.77%</b>) — proving
        why hand-written rules alone cannot replace ML.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Mini comparison table
        mini = df_compare[['Recall', 'F1 Score']].copy()
        mini = mini.reset_index()
        mini.columns = ['Model', 'Recall', 'F1']
        st.dataframe(
            mini.style
            .background_gradient(subset=['Recall'], cmap='Greens')
            .background_gradient(subset=['F1'], cmap='Blues')
            .format({'Recall': '{:.3f}', 'F1': '{:.3f}'}),
            hide_index=True,
            use_container_width=True
        )


# ─────────────────────────────────────────────────────────────
#  PAGE 2 — MODEL COMPARISON
# ─────────────────────────────────────────────────────────────
elif page == "📊 Model Comparison":
    st.markdown("""
    <h1 style='color:#ccd6f6; margin-bottom:4px;'>Model Comparison</h1>
    <p style='color:#8892b0; font-size:14px; margin-bottom:24px;'>
    Detailed breakdown of all 4 ML models + Rule-Based system
    </p>
    """, unsafe_allow_html=True)

    # Full metrics table
    st.markdown("<div class='section-header'>Complete Metrics Table</div>", unsafe_allow_html=True)
    display_df = df_compare.copy().reset_index()
    display_df.columns = [c.replace('_',' ') for c in display_df.columns]

    st.dataframe(
        display_df.style
        .background_gradient(subset=['Accuracy','Precision','Recall','F1 Score'], cmap='YlGn')
        .format({'Accuracy':'{:.4f}','Precision':'{:.4f}',
                 'Recall':'{:.4f}','F1 Score':'{:.4f}'}),
        hide_index=True,
        use_container_width=True,
        height=220
    )

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-header'>Recall Comparison</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor('#0f1117')
        ax.set_facecolor('#1a1d2e')

        models_list = df_compare.index.tolist()
        recall_vals = df_compare['Recall'].tolist()
        bar_colors  = ['#ff6b9d' if v == max(recall_vals) else '#2d3561' for v in recall_vals]

        bars = ax.barh(models_list, recall_vals, color=bar_colors,
                       edgecolor='none', height=0.55, zorder=3)
        for bar, val in zip(bars, recall_vals):
            ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=10, color='#ccd6f6', fontweight='bold')

        ax.set_xlim(0, 1.1)
        ax.set_xlabel('Recall Score', color='#8892b0')
        ax.tick_params(colors='#ccd6f6', labelsize=10)
        ax.grid(axis='x', color='#2d3561', alpha=0.5, zorder=0)
        ax.spines[['top','right','left','bottom']].set_color('#2d3561')
        ax.set_title('Recall — Who Catches the Most Churners?',
                     color='#ccd6f6', fontsize=11, fontweight='bold')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("<div class='section-header'>ROC-AUC Comparison</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor('#0f1117')
        ax.set_facecolor('#1a1d2e')

        auc_df = df_compare[df_compare['ROC-AUC'].notna()].copy()
        auc_vals  = auc_df['ROC-AUC'].astype(float).tolist()
        auc_names = auc_df.index.tolist()
        bar_colors = ['#64ffda' if v == max(auc_vals) else '#2d3561' for v in auc_vals]

        bars = ax.barh(auc_names, auc_vals, color=bar_colors,
                       edgecolor='none', height=0.55, zorder=3)
        for bar, val in zip(bars, auc_vals):
            ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=10, color='#ccd6f6', fontweight='bold')

        ax.set_xlim(0.75, 0.88)
        ax.set_xlabel('ROC-AUC Score', color='#8892b0')
        ax.tick_params(colors='#ccd6f6', labelsize=10)
        ax.grid(axis='x', color='#2d3561', alpha=0.5, zorder=0)
        ax.spines[['top','right','left','bottom']].set_color('#2d3561')
        ax.set_title('ROC-AUC — Overall Model Quality',
                     color='#ccd6f6', fontsize=11, fontweight='bold')
        st.pyplot(fig)
        plt.close()

    # Spider / Radar chart
    st.markdown("<div class='section-header'>Radar Chart — ML Models Only</div>", unsafe_allow_html=True)

    ml_models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost']
    metrics   = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    angles    = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles   += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#1a1d2e')

    radar_colors = ['#64ffda', '#ff6b9d', '#ffd166', '#a78bfa']

    for model_name, color in zip(ml_models, radar_colors):
        if model_name in df_compare.index:
            values = df_compare.loc[model_name, metrics].tolist()
            values += values[:1]
            ax.plot(angles, values, color=color, linewidth=2, label=model_name)
            ax.fill(angles, values, color=color, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, color='#ccd6f6', fontsize=11)
    ax.set_ylim(0, 1)
    ax.tick_params(colors='#8892b0')
    ax.grid(color='#2d3561', alpha=0.5)
    ax.spines['polar'].set_color('#2d3561')
    ax.set_facecolor('#1a1d2e')
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15),
              fontsize=9, labelcolor='#ccd6f6',
              facecolor='#1a1d2e', edgecolor='#2d3561')
    ax.set_title('Model Performance Radar', color='#ccd6f6',
                 fontsize=12, fontweight='bold', pad=20)

    col_center = st.columns([1, 2, 1])
    with col_center[1]:
        st.pyplot(fig)
    plt.close()


# ─────────────────────────────────────────────────────────────
#  PAGE 3 — LIVE PREDICTION
# ─────────────────────────────────────────────────────────────
elif page == "🔮 Live Prediction":
    st.markdown("""
    <h1 style='color:#ccd6f6; margin-bottom:4px;'>Live Churn Prediction</h1>
    <p style='color:#8892b0; font-size:14px; margin-bottom:24px;'>
    Enter customer details — all 5 models predict simultaneously
    </p>
    """, unsafe_allow_html=True)

    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.markdown("<div class='section-header'>Customer Details</div>", unsafe_allow_html=True)

        tenure      = st.slider("Tenure (months with company)", 0, 72, 12)
        income      = st.slider("Monthly Charges ($)", 18, 120, 65)
        purchases   = st.slider("Number of Active Services", 1, 9, 3)
        is_senior   = st.selectbox("Senior Citizen?", ["No", "Yes"])
        membership  = st.selectbox("Contract Type (Membership)",
                                   ["Month-to-month", "One year", "Two year"])
        gender      = st.selectbox("Gender", ["Male", "Female"])
        partner     = st.selectbox("Has Partner?", ["Yes", "No"])
        paperless   = st.selectbox("Paperless Billing?", ["Yes", "No"])

        predict_btn = st.button("🔮 Predict Churn", use_container_width=True)

    with col_result:
        st.markdown("<div class='section-header'>Prediction Results</div>", unsafe_allow_html=True)

        if predict_btn:
            # Rule-based prediction
            senior_val = 1 if is_senior == "Yes" else 0
            rule_pred, rule_reason = rule_based_predict(
                tenure, income, purchases, senior_val, membership
            )

            # ── FIXED: Build input — column order must exactly match training data ──
            input_df = pd.DataFrame([{
                'Age'             : senior_val,
                'Income'          : income,
                'Purchases'       : purchases,
                'tenure'          : tenure,
                'TotalCharges'    : income * tenure,
                'gender'          : 1 if gender == "Male" else 0,
                'Partner'         : 1 if partner == "Yes" else 0,
                'Dependents'      : 0,
                'PaperlessBilling': 1 if paperless == "Yes" else 0,
                'Membership_One year' : 1 if membership == "One year" else 0,
                'Membership_Two year' : 1 if membership == "Two year" else 0,
                'PaymentMethod_Credit card (automatic)': 0,
                'PaymentMethod_Electronic check'       : 1,
                'PaymentMethod_Mailed check'           : 0,
                'charges_per_service': income / (purchases + 1),
                'is_senior'          : senior_val,
            }])

            # Reorder columns to exactly match training order
            try:
                expected_cols = scaler.feature_names_in_
                input_df = input_df.reindex(columns=expected_cols, fill_value=0)
            except Exception:
                pass  # if scaler doesn't have feature_names_in_, use as-is

            # Predict with all ML models
            results_pred = {'Rule-Based': (rule_pred, None)}

            if scaler and models:
                try:
                    input_scaled = scaler.transform(input_df)
                    for name, model in models.items():
                        pred  = model.predict(input_scaled)[0]
                        proba = model.predict_proba(input_scaled)[0][1]
                        results_pred[name] = (pred, proba)
                except Exception as e:
                    st.warning(f"ML models couldn't run: {e}. Showing rule-based only.")

            # Display results
            churn_count = sum(1 for p, _ in results_pred.values() if p == 1)
            total_count = len(results_pred)

            if churn_count >= 3:
                st.markdown(f"""
                <div class='risk-high'>
                    <div style='font-size:42px;'>⚠️</div>
                    <div style='font-family:Syne; font-size:22px; font-weight:800;
                                color:#ff6b6b; margin:8px 0;'>HIGH CHURN RISK</div>
                    <div style='color:#8892b0; font-size:13px;'>
                    {churn_count} out of {total_count} models predict churn
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='risk-low'>
                    <div style='font-size:42px;'>✅</div>
                    <div style='font-family:Syne; font-size:22px; font-weight:800;
                                color:#64ffda; margin:8px 0;'>LOW CHURN RISK</div>
                    <div style='color:#8892b0; font-size:13px;'>
                    {churn_count} out of {total_count} models predict churn
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Per-model results
            for model_name, (pred, proba) in results_pred.items():
                label  = "🔴 Churn" if pred == 1 else "🟢 Stay"
                color  = "#ff6b6b" if pred == 1 else "#64ffda"
                prob_str = f"{proba*100:.1f}%" if proba is not None else "N/A"

                st.markdown(f"""
                <div style='display:flex; justify-content:space-between; align-items:center;
                            background:#1a1d2e; border:1px solid #2d3561; border-radius:10px;
                            padding:12px 18px; margin:6px 0;'>
                    <div style='font-family:Syne; font-weight:700;
                                color:#ccd6f6; font-size:13px;'>{model_name}</div>
                    <div style='display:flex; gap:16px; align-items:center;'>
                        <div style='font-size:11px; color:#8892b0;'>Probability: <b style='color:#ffd166;'>{prob_str}</b></div>
                        <div style='font-weight:700; color:{color}; font-size:13px;'>{label}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Rule reason
            st.markdown(f"""
            <div style='margin-top:14px; background:#1a1d2e; border:1px solid #2d356199;
                        border-radius:10px; padding:14px 18px;'>
                <div style='font-size:11px; color:#556; text-transform:uppercase;
                            letter-spacing:1px; margin-bottom:6px;'>Rule-Based Reason</div>
                <div style='font-size:13px; color:#8892b0;'>→ {rule_reason}</div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style='text-align:center; padding:60px 20px; color:#445;'>
                <div style='font-size:48px; margin-bottom:12px;'>🔮</div>
                <div style='font-size:14px;'>Adjust the sliders and click<br>
                <b style='color:#64ffda;'>Predict Churn</b> to see results</div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  PAGE 4 — RULE-BASED LOGIC
# ─────────────────────────────────────────────────────────────
elif page == "📋 Rule-Based Logic":
    st.markdown("""
    <h1 style='color:#ccd6f6; margin-bottom:4px;'>Rule-Based Churn Logic</h1>
    <p style='color:#8892b0; font-size:14px; margin-bottom:24px;'>
    Understanding the hand-crafted rules and how they compare to ML
    </p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("<div class='section-header'>The Rules I Built</div>", unsafe_allow_html=True)

        rules = [
            ("Rule 1 — Short Tenure Risk",
             "if month_to_month AND tenure < 12 → CHURN",
             "New customers on flexible contracts have the highest exit rate. No long-term commitment + short history = easy to leave."),
            ("Rule 2 — Poor Value Perception",
             "if income > $70 AND purchases ≤ 2 → CHURN",
             "Paying a lot for very few services. Customer likely feels they're not getting value for money."),
            ("Rule 3 — Senior + Short Contract",
             "if is_senior AND month_to_month → CHURN",
             "Senior customers on short-term plans showed elevated churn risk in EDA — often overwhelmed or seeking simpler options."),
            ("Rule 4 — Extended Month-to-Month",
             "if month_to_month AND tenure < 24 → CHURN",
             "Even slightly longer-tenured customers on flexible plans remain at risk until they commit to a longer contract."),
            ("Rule 5 — High Charge Ratio",
             "if charges_per_service > 25 AND NOT two_year → CHURN",
             "High cost per active service without a long-term commitment is a strong dissatisfaction signal."),
        ]

        for title, rule_code, explanation in rules:
            st.markdown(f"""
            <div style='background:#1a1d2e; border:1px solid #2d3561; border-radius:12px;
                        padding:16px; margin:10px 0;'>
                <div style='font-family:Syne; font-weight:700; color:#64ffda;
                            font-size:13px; margin-bottom:8px;'>{title}</div>
                <div style='background:#0f1117; border-radius:6px; padding:10px 14px;
                            font-family:monospace; font-size:12px; color:#ffd166;
                            margin-bottom:10px;'>{rule_code}</div>
                <div style='font-size:12px; color:#8892b0; line-height:1.6;'>{explanation}</div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='section-header'>Rule-Based vs ML Performance</div>", unsafe_allow_html=True)

        # Comparison bar chart
        metrics_list  = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        rule_vals     = [0.7393, 0.6087, 0.0377, 0.0711]
        xgb_vals      = [0.7229, 0.4858, 0.8302, 0.6129]
        rf_vals       = [0.7607, 0.5337, 0.7466, 0.6225]

        x      = np.arange(len(metrics_list))
        width  = 0.28

        fig, ax = plt.subplots(figsize=(7, 4.5))
        fig.patch.set_facecolor('#0f1117')
        ax.set_facecolor('#1a1d2e')

        b1 = ax.bar(x - width, rule_vals, width, label='Rule-Based',
                    color='#ff6b9d', alpha=0.85, zorder=3)
        b2 = ax.bar(x,          rf_vals,   width, label='Random Forest',
                    color='#ffd166', alpha=0.85, zorder=3)
        b3 = ax.bar(x + width,  xgb_vals,  width, label='XGBoost',
                    color='#64ffda', alpha=0.85, zorder=3)

        for bars in [b1, b2, b3]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                        f'{h:.2f}', ha='center', va='bottom',
                        fontsize=8, color='#8892b0')

        ax.set_xticks(x)
        ax.set_xticklabels(metrics_list, color='#ccd6f6', fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Score', color='#8892b0')
        ax.tick_params(colors='#8892b0')
        ax.legend(fontsize=9, labelcolor='#ccd6f6',
                  facecolor='#1a1d2e', edgecolor='#2d3561')
        ax.grid(axis='y', color='#2d3561', alpha=0.5, zorder=0)
        ax.spines[['top','right','left','bottom']].set_color('#2d3561')
        ax.set_title('Rule-Based vs Best ML Models', color='#ccd6f6',
                     fontsize=11, fontweight='bold')
        st.pyplot(fig)
        plt.close()

        # Key insight boxes
        st.markdown("""
        <div style='background:#1a1d2e; border:1px solid #ff6b9d44; border-radius:10px;
                    padding:16px; margin:12px 0;'>
            <div style='font-family:Syne; font-weight:700; color:#ff6b9d;
                        font-size:13px; margin-bottom:8px;'>⚠️ Rule-Based Weakness</div>
            <div style='font-size:12px; color:#8892b0; line-height:1.7;'>
            Recall of only <b style='color:#ff6b9d;'>3.77%</b> — it catches almost
            no actual churners. The rules are too conservative and miss the complex
            combinations of risk factors that ML naturally learns.
            </div>
        </div>

        <div style='background:#1a1d2e; border:1px solid #64ffda44; border-radius:10px;
                    padding:16px; margin:12px 0;'>
            <div style='font-family:Syne; font-weight:700; color:#64ffda;
                        font-size:13px; margin-bottom:8px;'>✅ When Rules Are Useful</div>
            <div style='font-size:12px; color:#8892b0; line-height:1.7;'>
            Rules have the highest Precision (<b style='color:#64ffda;'>0.608</b>) —
            when they flag someone, they're usually right. Useful for
            quick business decisions without needing a trained model,
            or for explaining decisions to non-technical stakeholders.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:#2d3561; margin-top:40px;'>
<div style='text-align:center; color:#445; font-size:11px; padding:10px 0 20px 0;'>
Built for AI & ML Internship Assessment · IBM Telco Churn Dataset · 4 ML Models + Rule-Based Logic
</div>
""", unsafe_allow_html=True)

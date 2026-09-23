"""
Credit Scoring — Streamlit Frontend
=====================================
Multi-page dashboard:
  🏠 Home           — project overview and model performance
  🔮 Single Predict — predict credit score for one customer
  📦 Batch Predict  — upload CSV and predict for many customers
  📊 EDA Gallery    — browse the generated visualisation charts
  ℹ️  About          — feature guide and API docs
"""

import io
import os
import sys
import json
import requests
import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE  = os.environ.get("API_BASE_URL", "http://localhost:5000")
BASE_DIR  = Path(__file__).resolve().parent.parent
FIG_DIR   = BASE_DIR / "outputs" / "figures"
META_JSON = BASE_DIR / "outputs" / "models" / "model_metadata.json"

st.set_page_config(
    page_title="Credit Scoring Model",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour palette ────────────────────────────────────────────────────────────
COLORS = {"Good": "#2ecc71", "Standard": "#f39c12", "Poor": "#e74c3c"}
BADGE  = {
    "Good":     "background:#2ecc71;color:white;padding:4px 14px;border-radius:20px;font-weight:600;font-size:1.1em",
    "Standard": "background:#f39c12;color:white;padding:4px 14px;border-radius:20px;font-weight:600;font-size:1.1em",
    "Poor":     "background:#e74c3c;color:white;padding:4px 14px;border-radius:20px;font-weight:600;font-size:1.1em",
}

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Page background */
    [data-testid="stAppViewContainer"] { background: #f0f4f8; }
    [data-testid="stSidebar"]          { background: #1a3a5c; }
    [data-testid="stSidebar"] * { color: #e8edf2 !important; }
    [data-testid="stSidebar"] .stRadio label { color: #e8edf2 !important; }

    /* Cards */
    .cs-card {
        background: white; border-radius: 12px;
        padding: 20px 24px; margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        border-left: 5px solid #3b82d4;
    }
    .cs-card-good     { border-left-color: #2ecc71; }
    .cs-card-standard { border-left-color: #f39c12; }
    .cs-card-poor     { border-left-color: #e74c3c; }

    /* Metric tiles */
    .metric-tile {
        background: white; border-radius: 10px; padding: 16px;
        text-align: center; box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }
    .metric-tile .val { font-size: 2rem; font-weight: 700; color: #1a3a5c; }
    .metric-tile .lbl { font-size: 0.82rem; color: #6b7280; margin-top: 2px; }

    /* Result box */
    .result-box {
        border-radius: 14px; padding: 28px 32px; text-align: center;
        margin: 16px 0; box-shadow: 0 4px 16px rgba(0,0,0,0.10);
    }

    /* Progress bars */
    .prob-bar-wrap { margin: 6px 0; }
    .prob-label    { font-size: 0.85rem; font-weight: 600; margin-bottom: 2px; }

    /* Table */
    .styled-table { width:100%; border-collapse:collapse; font-size:0.9em; }
    .styled-table th {
        background:#1a3a5c; color:white; padding:10px 14px; text-align:left;
    }
    .styled-table td { padding:9px 14px; border-bottom:1px solid #e5e7eb; }
    .styled-table tr:nth-child(even) td { background:#f7f8fa; }
    .best-row td { font-weight:700; background:#fffbeb !important; }

    /* Sidebar nav */
    .nav-title { font-size:1.2rem; font-weight:700; padding:12px 0 6px; }
    div[data-testid="stRadio"] > label { font-size:0.97rem !important; }

    h1 { color: #1a3a5c; }
    h2 { color: #1a3a5c; }
    h3 { color: #2c5282; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def api(path, method="GET", payload=None, timeout=15):
    """Call the Flask API; return (data_dict, error_str)."""
    url = f"{API_BASE}{path}"
    try:
        if method == "GET":
            r = requests.get(url, timeout=timeout)
        else:
            r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, f"Cannot connect to API at {API_BASE}. Is the Flask server running?"
    except requests.exceptions.HTTPError as e:
        try:
            msg = r.json().get("error", str(e))
        except Exception:
            msg = str(e)
        return None, msg
    except Exception as e:
        return None, str(e)


def badge(label):
    return f'<span style="{BADGE.get(label, BADGE["Standard"])}">{label}</span>'


def prob_bar(label, value, color):
    pct = int(value * 100)
    return f"""
    <div class="prob-bar-wrap">
      <div class="prob-label">{label}</div>
      <div style="background:#e5e7eb;border-radius:6px;height:20px;overflow:hidden">
        <div style="width:{pct}%;background:{color};height:100%;
                    display:flex;align-items:center;padding-left:6px;
                    color:white;font-size:0.78rem;font-weight:600;
                    transition:width 0.4s ease">
          {pct}%
        </div>
      </div>
    </div>"""


def load_metadata():
    if META_JSON.exists():
        with open(META_JSON) as f:
            return json.load(f)
    return {}


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 💳 Credit Scoring")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔮 Single Predict", "📦 Batch Predict",
         "📊 EDA Gallery", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    # API status
    health, err = api("/health")
    if health:
        st.success("🟢 API Online")
        st.caption(f"Model: {health.get('model', '—')}")
    else:
        st.error("🔴 API Offline")
        st.caption("Start: `python backend/app.py`")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("💳 Credit Scoring Model Dashboard")
    st.markdown("Predict customer creditworthiness as **Good**, **Standard**, or **Poor** "
                "using a Random Forest model trained on 100,000 financial records.")

    st.markdown("---")

    # KPI tiles
    meta = load_metadata()
    best = meta.get("metrics", {}).get("Random Forest", {})
    c1, c2, c3, c4, c5 = st.columns(5)
    tiles = [
        (c1, f"{best.get('Accuracy', 0):.1%}", "Accuracy"),
        (c2, f"{best.get('F1-Score', 0):.4f}", "F1-Score"),
        (c3, f"{best.get('ROC-AUC', 0):.4f}", "ROC-AUC"),
        (c4, "100K", "Training Records"),
        (c5, "50K",  "Test Predictions"),
    ]
    for col, val, lbl in tiles:
        col.markdown(
            f'<div class="metric-tile"><div class="val">{val}</div>'
            f'<div class="lbl">{lbl}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.subheader("📈 Model Performance")
        metrics_data, err = api("/metrics")
        if metrics_data:
            rows_html = ""
            for m in metrics_data["models"]:
                best_cls = "best-row" if m["is_best"] else ""
                star     = " ⭐" if m["is_best"] else ""
                rows_html += f"""
                <tr class="{best_cls}">
                  <td>{m['model']}{star}</td>
                  <td>{m['accuracy']:.4f}</td>
                  <td>{m['precision']:.4f}</td>
                  <td>{m['recall']:.4f}</td>
                  <td>{m['f1_score']:.4f}</td>
                  <td>{m['roc_auc']:.4f}</td>
                </tr>"""
            st.markdown(f"""
            <table class="styled-table">
              <tr><th>Model</th><th>Accuracy</th><th>Precision</th>
                  <th>Recall</th><th>F1-Score</th><th>ROC-AUC</th></tr>
              {rows_html}
            </table>""", unsafe_allow_html=True)
        else:
            st.warning(f"Cannot load metrics — {err}")

    with col_r:
        st.subheader("🎯 Target Distribution (Train)")
        fig_path = FIG_DIR / "01_target_distribution.png"
        if fig_path.exists():
            st.image(str(fig_path), use_container_width=True)

    st.markdown("---")
    st.subheader("🔑 Key Insights")
    insights = [
        ("💰 Outstanding Debt", "Good customers average **$801** vs **$2,081** for Poor — a 2.6× gap."),
        ("📅 Credit History",   "Longer credit history strongly correlates with **Good** credit score."),
        ("⏰ Payment Delays",   "Poor customers average **32 delayed payments** vs **26** for Good."),
        ("💳 Credit Mix",       "Customers with 'Good' credit mix have **~60% chance** of Good score."),
        ("📊 Engineered Features", "Debt-to-Income & EMI-to-Salary ratios rank in **top 5 features**."),
    ]
    cols = st.columns(len(insights))
    for col, (title, body) in zip(cols, insights):
        col.markdown(
            f'<div class="cs-card"><strong>{title}</strong><br><small>{body}</small></div>',
            unsafe_allow_html=True,
        )


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SINGLE PREDICTION
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Single Predict":
    st.title("🔮 Predict Credit Score")
    st.markdown("Fill in the customer financial details and click **Predict**.")

    with st.form("predict_form"):
        st.subheader("👤 Personal & Employment")
        c1, c2, c3 = st.columns(3)
        age        = c1.number_input("Age",          min_value=18, max_value=100, value=34)
        occupation = c2.selectbox("Occupation", [
            "Accountant","Architect","Developer","Doctor","Engineer","Entrepreneur",
            "Journalist","Lawyer","Manager","Mechanic","Media_Manager","Musician",
            "Scientist","Teacher","Writer",
        ])
        annual_income = c3.number_input("Annual Income ($)", min_value=0.0, value=50000.0, step=1000.0)

        st.subheader("💵 Income & Balance")
        c1, c2, c3 = st.columns(3)
        monthly_salary  = c1.number_input("Monthly Inhand Salary ($)", min_value=0.0, value=3500.0, step=100.0)
        monthly_balance = c2.number_input("Monthly Balance ($)",        min_value=0.0, value=350.0,  step=50.0)
        amount_invested = c3.number_input("Amount Invested Monthly ($)",min_value=0.0, value=150.0,  step=10.0)

        st.subheader("🏦 Credit Profile")
        c1, c2, c3, c4 = st.columns(4)
        num_bank_acc  = c1.number_input("Bank Accounts",    min_value=0, max_value=50,  value=3)
        num_cc        = c2.number_input("Credit Cards",     min_value=0, max_value=20,  value=4)
        interest_rate = c3.number_input("Interest Rate (%)",min_value=0.0, max_value=50.0, value=12.0, step=0.5)
        num_loan      = c4.number_input("Number of Loans",  min_value=0, max_value=20,  value=2)

        c1, c2, c3 = st.columns(3)
        outstanding_debt  = c1.number_input("Outstanding Debt ($)",          min_value=0.0, value=1200.0, step=100.0)
        credit_util       = c2.number_input("Credit Utilization Ratio (%)",  min_value=0.0, max_value=100.0, value=28.5, step=0.5)
        changed_cl        = c3.number_input("Changed Credit Limit ($)",      min_value=0.0, value=500.0, step=50.0)

        c1, c2 = st.columns(2)
        credit_history_age = c1.text_input("Credit History Age", value="5 Years and 3 Months",
                                           help="Format: 'X Years and Y Months'")
        num_credit_inq     = c2.number_input("Credit Inquiries", min_value=0, max_value=50, value=4)

        st.subheader("📋 Payment Behaviour")
        c1, c2, c3 = st.columns(3)
        credit_mix        = c1.selectbox("Credit Mix",              ["Good","Standard","Bad"])
        payment_min       = c2.selectbox("Pays Min Amount?",        ["Yes","No","NM"])
        payment_behaviour = c3.selectbox("Payment Behaviour", [
            "Low_spent_Small_value_payments","Low_spent_Medium_value_payments",
            "Low_spent_Large_value_payments","High_spent_Small_value_payments",
            "High_spent_Medium_value_payments","High_spent_Large_value_payments",
        ])

        c1, c2, c3 = st.columns(3)
        delay_due_date  = c1.number_input("Days Delay from Due Date", min_value=-10, max_value=100, value=5)
        num_delayed_pay = c2.number_input("Num of Delayed Payments",  min_value=0, max_value=50,   value=3)
        total_emi       = c3.number_input("Total EMI per Month ($)",  min_value=0.0, value=200.0, step=10.0)
        num_loan_types  = st.number_input("Number of Loan Types",    min_value=0, max_value=10, value=2)

        submitted = st.form_submit_button("🔮 Predict Credit Score", use_container_width=True)

    if submitted:
        record = {
            "Age":                    age,
            "Occupation":             occupation,
            "Annual_Income":          annual_income,
            "Monthly_Inhand_Salary":  monthly_salary,
            "Num_Bank_Accounts":      num_bank_acc,
            "Num_Credit_Card":        num_cc,
            "Interest_Rate":          interest_rate,
            "Num_of_Loan":            num_loan,
            "Delay_from_due_date":    delay_due_date,
            "Num_of_Delayed_Payment": num_delayed_pay,
            "Changed_Credit_Limit":   changed_cl,
            "Num_Credit_Inquiries":   num_credit_inq,
            "Credit_Mix":             credit_mix,
            "Outstanding_Debt":       outstanding_debt,
            "Credit_Utilization_Ratio": credit_util,
            "Payment_of_Min_Amount":  payment_min,
            "Total_EMI_per_month":    total_emi,
            "Amount_invested_monthly":amount_invested,
            "Payment_Behaviour":      payment_behaviour,
            "Monthly_Balance":        monthly_balance,
            "Credit_History_Age":     credit_history_age,
            "Num_Loan_Types":         num_loan_types,
        }

        with st.spinner("Running prediction…"):
            result, err = api("/predict", method="POST", payload=record)

        if err:
            st.error(f"Prediction error: {err}")
        else:
            pred = result["prediction"]
            label = pred["credit_score"]
            color = pred["color"]
            conf  = pred["confidence"]
            probs = pred["probabilities"]

            st.markdown(f"""
            <div class="result-box" style="background:{color}20;border:2px solid {color}">
              <div style="font-size:3rem">{ {"Good":"✅","Standard":"⚠️","Poor":"❌"}[label] }</div>
              <div style="font-size:2rem;font-weight:700;color:{color}">{label}</div>
              <div style="color:#4b5563;margin-top:4px">
                Confidence: <strong>{conf:.1%}</strong>
              </div>
            </div>""", unsafe_allow_html=True)

            st.markdown("#### Probability Breakdown")
            for lbl, clr in [("Good","#2ecc71"),("Standard","#f39c12"),("Poor","#e74c3c")]:
                st.markdown(prob_bar(lbl, probs[lbl], clr), unsafe_allow_html=True)

            # Risk summary
            st.markdown("#### Risk Summary")
            risk_items = []
            if outstanding_debt > 2000:
                risk_items.append("⚠️ High outstanding debt (> $2,000)")
            if num_delayed_pay > 10:
                risk_items.append("⚠️ Many delayed payments (> 10)")
            if credit_util > 40:
                risk_items.append("⚠️ High credit utilisation (> 40%)")
            if total_emi / (monthly_salary + 1e-6) > 0.4:
                risk_items.append("⚠️ EMI-to-salary ratio exceeds 40%")
            if credit_mix == "Bad":
                risk_items.append("⚠️ Poor credit mix")
            if not risk_items:
                st.success("✅ No major risk flags detected.")
            else:
                for item in risk_items:
                    st.warning(item)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — BATCH PREDICTION
# ════════════════════════════════════════════════════════════════════════════
elif page == "📦 Batch Predict":
    st.title("📦 Batch Credit Score Predictions")
    st.markdown("Upload a CSV file with customer records to generate predictions for all rows.")

    st.info(
        "**Required columns (minimum):** Age, Annual_Income, Monthly_Inhand_Salary, "
        "Num_Bank_Accounts, Num_Credit_Card, Interest_Rate, Outstanding_Debt, "
        "Credit_Utilization_Ratio, Delay_from_due_date, Num_of_Delayed_Payment, "
        "Total_EMI_per_month, Credit_Mix, Payment_of_Min_Amount, Payment_Behaviour"
    )

    # Sample download
    sample = pd.DataFrame([{
        "Age": 34, "Annual_Income": 50000, "Monthly_Inhand_Salary": 3500,
        "Num_Bank_Accounts": 3, "Num_Credit_Card": 4, "Interest_Rate": 12,
        "Num_of_Loan": 2, "Delay_from_due_date": 5, "Num_of_Delayed_Payment": 3,
        "Changed_Credit_Limit": 500, "Num_Credit_Inquiries": 4,
        "Outstanding_Debt": 1200, "Credit_Utilization_Ratio": 28.5,
        "Credit_History_Age": "5 Years and 3 Months",
        "Total_EMI_per_month": 200, "Amount_invested_monthly": 150,
        "Monthly_Balance": 350, "Num_Loan_Types": 2,
        "Occupation": "Engineer", "Credit_Mix": "Good",
        "Payment_of_Min_Amount": "Yes",
        "Payment_Behaviour": "Low_spent_Medium_value_payments",
    }])
    st.download_button(
        "⬇️ Download Sample CSV",
        data=sample.to_csv(index=False),
        file_name="sample_customers.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])

    if uploaded:
        df_upload = pd.read_csv(uploaded)
        st.markdown(f"**Loaded:** {len(df_upload):,} rows × {len(df_upload.columns)} columns")
        st.dataframe(df_upload.head(5), use_container_width=True)

        if st.button("🚀 Run Batch Prediction", use_container_width=True):
            records = df_upload.to_dict(orient="records")

            # Split into chunks of 500
            all_predictions = []
            chunks = [records[i:i+500] for i in range(0, len(records), 500)]
            progress = st.progress(0)

            for i, chunk in enumerate(chunks):
                result, err = api("/predict/batch", method="POST", payload=chunk)
                if err:
                    st.error(f"Batch error on chunk {i+1}: {err}")
                    break
                all_predictions.extend(result["predictions"])
                progress.progress((i + 1) / len(chunks))

            progress.empty()

            if all_predictions:
                pred_df = pd.DataFrame([{
                    "Row":          p["index"] + 1,
                    "Credit_Score": p["credit_score"],
                    "Confidence":   f"{p['confidence']:.1%}",
                    "Prob_Poor":    p["probabilities"]["Poor"],
                    "Prob_Standard":p["probabilities"]["Standard"],
                    "Prob_Good":    p["probabilities"]["Good"],
                } for p in all_predictions])

                # Summary
                summary = pred_df["Credit_Score"].value_counts()
                st.subheader("📊 Prediction Summary")
                c1, c2, c3 = st.columns(3)
                for col, label, color in [(c1,"Good","#2ecc71"),(c2,"Standard","#f39c12"),(c3,"Poor","#e74c3c")]:
                    n = summary.get(label, 0)
                    col.markdown(
                        f'<div class="metric-tile" style="border-top:4px solid {color}">'
                        f'<div class="val" style="color:{color}">{n:,}</div>'
                        f'<div class="lbl">{label}</div></div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("---")
                st.subheader("📋 Predictions Table")
                st.dataframe(pred_df, use_container_width=True, height=400)

                # Merge with original for download
                result_df = df_upload.copy()
                result_df["Predicted_Credit_Score"] = [p["credit_score"] for p in all_predictions]
                result_df["Confidence"]             = [p["confidence"]   for p in all_predictions]
                result_df["Prob_Poor"]              = [p["probabilities"]["Poor"]     for p in all_predictions]
                result_df["Prob_Standard"]          = [p["probabilities"]["Standard"] for p in all_predictions]
                result_df["Prob_Good"]              = [p["probabilities"]["Good"]     for p in all_predictions]

                st.download_button(
                    "⬇️ Download Predictions CSV",
                    data=result_df.to_csv(index=False),
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — EDA GALLERY
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA Gallery":
    st.title("📊 Exploratory Data Analysis Gallery")

    figures = {
        "01_target_distribution.png":           "Target Distribution — Credit Score Classes",
        "02_missing_values.png":                "Missing Values per Column",
        "03_numerical_distributions.png":       "Numerical Feature Distributions",
        "04_boxplots_by_credit_score.png":      "Key Features vs Credit Score (Boxplots)",
        "05_categorical_vs_credit_score.png":   "Categorical Features vs Credit Score",
        "06_correlation_heatmap.png":           "Correlation Heatmap",
        "07_feature_correlation_with_target.png":"Feature Correlations with Target",
        "08_income_by_credit_score.png":        "Annual Income by Credit Score",
        "09_delayed_payments_by_score.png":     "Delayed Payments by Credit Score",
        "10_outstanding_debt_by_score.png":     "Outstanding Debt by Credit Score",
        "11_monthly_balance_violin.png":        "Monthly Balance Violin Plot",
        "12_model_comparison.png":              "Model Performance Comparison",
        "13_confusion_matrices.png":            "Confusion Matrices (All Models)",
        "14_roc_curves.png":                    "ROC Curves (One-vs-Rest)",
        "15_feature_importance.png":            "Top 20 Feature Importances (Random Forest)",
        "16_test_predictions_distribution.png": "Predicted Distribution on Test Set",
    }

    # Filter selector
    cat_filter = st.selectbox("Filter by category", [
        "All", "EDA (01-11)", "Model Results (12-16)"
    ])
    filtered = {k: v for k, v in figures.items() if (
        cat_filter == "All" or
        (cat_filter == "EDA (01-11)"       and int(k[:2]) <= 11) or
        (cat_filter == "Model Results (12-16)" and int(k[:2]) >= 12)
    )}

    for i, (fname, caption) in enumerate(filtered.items()):
        path = FIG_DIR / fname
        if path.exists():
            if i % 2 == 0:
                cols = st.columns(2)
            cols[i % 2].image(str(path), caption=caption, use_container_width=True)
        else:
            st.warning(f"Figure not found: {fname}. Run `python scripts/analysis.py` first.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ABOUT
# ════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("ℹ️ About this Project")

    st.markdown("""
    ## Credit Scoring Model

    An end-to-end machine learning system that predicts customer **creditworthiness**
    from financial history data. Built with Python, Scikit-learn, Flask, and Streamlit.

    ---

    ### 🏗️ Architecture

    ```
    ┌──────────────────────┐        HTTP REST        ┌────────────────────────┐
    │   Streamlit Frontend │ ──────────────────────► │   Flask Backend API    │
    │   frontend/app.py    │ ◄────────────────────── │   backend/app.py       │
    │   Port: 8501         │    JSON responses        │   Port: 5000           │
    └──────────────────────┘                          └──────────┬─────────────┘
                                                                  │  loads
                                                       ┌──────────▼─────────────┐
                                                       │  Random Forest Model   │
                                                       │  outputs/models/       │
                                                       │  best_model.pkl        │
                                                       └────────────────────────┘
    ```

    ---

    ### 🔌 API Endpoints

    | Method | Endpoint | Description |
    |--------|----------|-------------|
    | GET | `/health` | Service health check |
    | GET | `/model/info` | Model metadata & features |
    | GET | `/features` | Input field descriptions |
    | GET | `/metrics` | All model comparison metrics |
    | POST | `/predict` | Predict single customer |
    | POST | `/predict/batch` | Predict batch (up to 500) |

    ---

    ### 📐 Feature Descriptions

    | Feature | Description |
    |---------|-------------|
    | Age | Customer age in years |
    | Annual_Income | Gross yearly income ($) |
    | Monthly_Inhand_Salary | Net monthly take-home pay ($) |
    | Num_Bank_Accounts | Active bank accounts count |
    | Num_Credit_Card | Credit cards held |
    | Interest_Rate | Average loan interest rate (%) |
    | Num_of_Loan | Active loans count |
    | Outstanding_Debt | Total outstanding debt ($) |
    | Credit_Utilization_Ratio | Credit usage as % of limit |
    | Credit_History_Age | Length of credit history |
    | Delay_from_due_date | Avg days payment was delayed |
    | Num_of_Delayed_Payment | Count of late payments |
    | Total_EMI_per_month | Monthly EMI burden ($) |
    | Amount_invested_monthly | Monthly investments ($) |
    | Monthly_Balance | End-of-month cash balance ($) |
    | Credit_Mix | Credit portfolio quality (Good/Standard/Bad) |
    | Payment_of_Min_Amount | Pays minimum amount? (Yes/No) |
    | Payment_Behaviour | Spending & payment pattern |
    | Occupation | Customer occupation |

    ---

    ### 🧪 Engineered Features

    | Feature | Formula |
    |---------|---------|
    | Debt_to_Income | Outstanding Debt ÷ Annual Income |
    | EMI_to_Salary | Total EMI ÷ Monthly Salary |
    | Savings_Rate | Amount Invested ÷ Monthly Salary |
    | Balance_Buffer | Monthly Balance ÷ Monthly Salary |
    | Cards_per_Account | Num Credit Cards ÷ Num Bank Accounts |
    | Credit_History_Months | Parsed from text to numeric months |
    | Num_Loan_Types | Count of distinct loan types |

    ---

    ### 🚀 Running the Application

    ```bash
    # 1. Install dependencies
    pip install -r requirements.txt

    # 2. Train the model (if not already done)
    python scripts/model.py

    # 3. Start Flask backend (Terminal 1)
    python backend/app.py

    # 4. Start Streamlit frontend (Terminal 2)
    streamlit run frontend/app.py
    ```

    Open **http://localhost:8501** in your browser.
    """)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Training Records",  "100,000")
    col2.metric("Features Used",     "27")
    col3.metric("Best Model",        "Random Forest")
    col1.metric("Accuracy",  "73.33%")
    col2.metric("F1-Score",  "0.7344")
    col3.metric("ROC-AUC",   "0.8583")

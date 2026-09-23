# 💳 Credit Scoring Model

> **End-to-end Machine Learning system** that predicts customer creditworthiness — **Good**, **Standard**, or **Poor** — from financial history data.  
> Includes a **Flask REST API** backend and a **Streamlit** interactive dashboard frontend.

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Architecture](#-architecture)
3. [Project Structure](#-project-structure)
4. [Quick Start](#-quick-start)
5. [Dataset](#-dataset)
6. [Feature Engineering](#-feature-engineering)
7. [Models & Results](#-models--results)
8. [Flask API Reference](#-flask-api-reference)
9. [Streamlit Dashboard](#-streamlit-dashboard)
10. [Key Insights](#-key-insights)
11. [Output Files](#-output-files)

---

## 🔍 Project Overview

| Item | Detail |
|------|--------|
| **Goal** | Classify customers as Good / Standard / Poor credit risk |
| **Training data** | 100,000 records × 28 features |
| **Test data** | 50,000 records (predictions generated) |
| **Best model** | Random Forest — F1 = 0.7344, ROC-AUC = 0.858 |
| **Stack** | Python · Scikit-learn · Flask · Streamlit · Pandas · Matplotlib |

---

## 🏗️ Architecture

```
┌──────────────────────────┐         HTTP REST         ┌─────────────────────────┐
│   Streamlit Frontend     │ ───────────────────────►  │   Flask Backend API     │
│   frontend/app.py        │ ◄───────────────────────  │   backend/app.py        │
│   http://localhost:8501  │       JSON responses       │   http://localhost:5000  │
└──────────────────────────┘                            └──────────┬──────────────┘
                                                                    │ loads
                                                         ┌──────────▼──────────────┐
                                                         │  Shared Preprocessor    │
                                                         │  backend/preprocessor.py│
                                                         └──────────┬──────────────┘
                                                                    │ uses
                                                         ┌──────────▼──────────────┐
                                                         │  Random Forest Model    │
                                                         │  outputs/models/        │
                                                         │  best_model.pkl         │
                                                         └─────────────────────────┘
```

---

## 📁 Project Structure

```
Credit Scoring Model/
│
├── 📂 data/
│   ├── train.csv                        # 100,000 training records
│   └── test.csv                         # 50,000 test records
│
├── 📂 backend/
│   ├── app.py                           # Flask REST API (6 endpoints)
│   ├── preprocessor.py                  # Shared preprocessing & feature engineering
│   └── test_api.py                      # API smoke tests
│
├── 📂 frontend/
│   └── app.py                           # Streamlit multi-page dashboard
│
├── 📂 notebooks/
│   └── credit_scoring_analysis.ipynb    # Complete Jupyter notebook (13 sections)
│
├── 📂 scripts/
│   ├── analysis.py                      # EDA script → 11 visualisation charts
│   ├── model.py                         # Feature engineering + 3 ML models
│   └── generate_report.py              # Word report generator
│
├── 📂 outputs/
│   ├── figures/                         # 16 PNG visualisation charts
│   ├── models/
│   │   ├── best_model.pkl               # Saved Random Forest model
│   │   └── model_metadata.json          # Model config & metrics
│   └── predictions/
│       ├── test_predictions.csv         # 50,000 predictions with probabilities
│       └── model_comparison.csv         # Model metrics comparison
│
├── 📂 reports/
│   └── credit_scoring_report.docx       # Full Word project report (1.6 MB)
│
├── start_backend.bat                    # One-click: start Flask API
├── start_frontend.bat                   # One-click: start Streamlit
├── README.md
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model *(skip if `outputs/models/best_model.pkl` already exists)*

```bash
python scripts/analysis.py   # EDA + figures
python scripts/model.py      # Training + predictions
```

### 3. Start the Flask backend

```bash
# Option A — double-click
start_backend.bat

# Option B — terminal
python backend/app.py
```

Backend available at **http://localhost:5000**

### 4. Start the Streamlit frontend

Open a **second** terminal:

```bash
# Option A — double-click
start_frontend.bat

# Option B — terminal
streamlit run frontend/app.py
```

Dashboard available at **http://localhost:8501**

### 5. Run API tests

```bash
python backend/test_api.py
```

### 6. Open the Jupyter notebook *(optional)*

```bash
jupyter notebook notebooks/credit_scoring_analysis.ipynb
```

---

## 📊 Dataset

The dataset contains monthly financial snapshots for **12,500 unique customers** tracked over **8 months**.

### Target variable

| Class | Count | Proportion |
|-------|-------|-----------|
| Standard | 53,174 | 53.2% |
| Poor | 28,998 | 29.0% |
| Good | 17,828 | 17.8% |

### Missing values

| Column | Missing | % |
|--------|---------|---|
| Monthly_Inhand_Salary | 15,002 | 15.0% |
| Type_of_Loan | 11,408 | 11.4% |
| Credit_History_Age | 9,030 | 9.0% |
| Num_of_Delayed_Payment | 7,002 | 7.0% |

All missing values are handled via **median imputation** (fit on train only — no leakage).

---

## 🔧 Feature Engineering

Seven features are engineered from raw financial data:

| Feature | Formula | Business Meaning |
|---------|---------|-----------------|
| `Debt_to_Income` | Outstanding Debt ÷ Annual Income | Overall debt burden |
| `EMI_to_Salary` | Total EMI ÷ Monthly Salary | Monthly repayment pressure |
| `Savings_Rate` | Amount Invested ÷ Monthly Salary | Financial discipline |
| `Balance_Buffer` | Monthly Balance ÷ Monthly Salary | Cash cushion |
| `Cards_per_Account` | Num Credit Cards ÷ Num Bank Accounts | Credit card concentration |
| `Credit_History_Months` | Parsed from text "X Years and Y Months" | Numeric credit age |
| `Num_Loan_Types` | Count of distinct loan types | Credit diversity |

### Preprocessing pipeline (no data leakage)

1. Drop identifiers (ID, Customer_ID, Name, SSN, Month)
2. Strip special characters from noisy numeric fields
3. Parse `Credit_History_Age` text → numeric months
4. Count loan types from comma-separated string
5. Engineer 5 financial ratio features
6. Label-encode all categorical features
7. Replace `inf`/`-inf` → `NaN`, impute with column median
8. `StandardScaler` — **fit on train only**, transform applied to both
9. Final `NaN`/`inf` safety pass after scaling
10. Stratified 80/20 train/validation split

---

## 🤖 Models & Results

Three models were trained and evaluated on the 20,000-record stratified validation set:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | CV-Acc |
|-------|----------|-----------|--------|----------|---------|--------|
| Logistic Regression | 59.90% | 59.59% | 59.90% | 58.18% | 0.737 | 60.29% |
| Decision Tree | 70.91% | 71.58% | 70.91% | 71.02% | 0.831 | 69.58% |
| **Random Forest ⭐** | **73.33%** | **73.73%** | **73.33%** | **73.44%** | **0.858** | **70.03%** |

**Random Forest** was selected, retrained on all 100,000 records, and used to generate test predictions.

---

## 🔌 Flask API Reference

Base URL: `http://localhost:5000`

### `GET /health`
Service health check.
```json
{ "status": "ok", "model": "Random Forest", "version": "1.0.0" }
```

### `GET /model/info`
Model metadata, features list, and metrics.

### `GET /features`
All input field descriptions, types, and allowed values.

### `GET /metrics`
Model comparison table (all 3 models).

### `POST /predict`
Predict credit score for a single customer.

**Request body:**
```json
{
  "Age": 34,
  "Annual_Income": 50000,
  "Monthly_Inhand_Salary": 3500,
  "Num_Bank_Accounts": 3,
  "Num_Credit_Card": 4,
  "Interest_Rate": 12,
  "Num_of_Loan": 2,
  "Delay_from_due_date": 5,
  "Num_of_Delayed_Payment": 3,
  "Changed_Credit_Limit": 500,
  "Num_Credit_Inquiries": 4,
  "Outstanding_Debt": 1200,
  "Credit_Utilization_Ratio": 28.5,
  "Credit_History_Age": "5 Years and 3 Months",
  "Total_EMI_per_month": 200,
  "Amount_invested_monthly": 150,
  "Monthly_Balance": 350,
  "Num_Loan_Types": 2,
  "Occupation": "Engineer",
  "Credit_Mix": "Good",
  "Payment_of_Min_Amount": "Yes",
  "Payment_Behaviour": "Low_spent_Medium_value_payments"
}
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "credit_score": "Standard",
    "confidence": 0.566,
    "probabilities": { "Poor": 0.202, "Standard": 0.566, "Good": 0.233 },
    "color": "#f39c12",
    "emoji": "⚠️"
  }
}
```

### `POST /predict/batch`
Predict for a list of customers (up to 500 per request).

**Request body:** JSON array of customer objects (same schema as `/predict`).

**Response:**
```json
{
  "success": true,
  "count": 3,
  "predictions": [ { "index": 0, "credit_score": "Good", ... }, ... ],
  "summary": { "Good": 1, "Standard": 2, "Poor": 0 }
}
```

---

## 🖥️ Streamlit Dashboard

The dashboard has **5 pages** accessible from the left sidebar:

| Page | Description |
|------|-------------|
| 🏠 **Home** | KPI tiles, model performance table, key insights, target distribution chart |
| 🔮 **Single Predict** | Interactive form to predict one customer's credit score with probability bars and risk flags |
| 📦 **Batch Predict** | Upload a CSV, run bulk predictions, download results |
| 📊 **EDA Gallery** | Browse all 16 generated visualisation charts with category filtering |
| ℹ️ **About** | Architecture diagram, feature guide, API reference, setup instructions |

---

## 💡 Key Insights

1. **Outstanding Debt is #1 predictor** — Good customers average **$801** vs **$2,081** for Poor (2.6× difference)
2. **Credit History Age** — longer history strongly predicts Good credit score
3. **Payment delays** — Poor customers average **32 delayed payments** vs **26** for Good
4. **Credit Mix** — customers with Good credit mix have **~60% chance** of Good credit score
5. **Income alone is insufficient** — income distributions overlap; debt management matters more
6. **Engineered features add value** — `EMI_to_Salary` and `Debt_to_Income` rank in top 5 by importance
7. **Data quality is critical** — raw data contains negative ages, extreme outliers, special characters requiring robust cleaning

---

## 📦 Output Files

| File | Size | Description |
|------|------|-------------|
| `outputs/predictions/test_predictions.csv` | ~4 MB | 50,000 predictions with class probabilities |
| `outputs/models/best_model.pkl` | ~60 MB | Serialised Random Forest model |
| `outputs/models/model_metadata.json` | ~1 KB | Model config, features, and metrics |
| `outputs/predictions/model_comparison.csv` | ~1 KB | All models' evaluation metrics |
| `outputs/figures/*.png` | ~1.6 MB total | 16 EDA and model evaluation charts |
| `reports/credit_scoring_report.docx` | ~1.6 MB | Full project report with embedded charts |

---

## 📦 Requirements

| Package | Purpose |
|---------|---------|
| pandas, numpy | Data manipulation |
| scikit-learn | ML models, preprocessing, evaluation |
| joblib | Model serialisation |
| matplotlib, seaborn | Visualisation |
| flask, flask-cors | REST API backend |
| streamlit | Interactive frontend dashboard |
| requests | Streamlit → Flask HTTP calls |
| python-docx | Word report generation |
| jupyter, ipykernel | Notebook support |

Install all with:
```bash
pip install -r requirements.txt
```

---

## 📜 License

This project is provided for educational and portfolio purposes.

"""
Credit Scoring Model — Generate Word Report (.docx)
====================================================
Creates a comprehensive project report with content and embedded figures.
"""

import warnings
warnings.filterwarnings("ignore")

from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from pathlib import Path
import datetime

FIG_DIR     = Path("outputs/figures")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

doc = Document()

# ── Page margins ────────────────────────────────────────────────────────────
section = doc.sections[0]
section.left_margin   = Cm(2.5)
section.right_margin  = Cm(2.5)
section.top_margin    = Cm(2.5)
section.bottom_margin = Cm(2.5)

# ── Helpers ──────────────────────────────────────────────────────────────────
def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = RGBColor(0x1a, 0x3a, 0x5c)
    return h

def add_para(doc, text, bold=False, size=11):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.font.size = Pt(size)
    return p

def add_figure(doc, path, caption="", width=6.0):
    if Path(path).exists():
        doc.add_picture(str(path), width=Inches(width))
        last = doc.paragraphs[-1]
        last.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if caption:
            cap = doc.add_paragraph(caption)
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in cap.runs:
                run.font.italic = True
                run.font.size   = Pt(9)
                run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    else:
        doc.add_paragraph(f"[Figure not found: {path}]")

def set_cell_bg(cell, color_hex):
    shading = OxmlElement("w:shd")
    shading.set(qn("w:val"), "clear")
    shading.set(qn("w:color"), "auto")
    shading.set(qn("w:fill"), color_hex)
    cell._tc.get_or_add_tcPr().append(shading)

def add_table(doc, headers, rows, header_color="1A3A5C"):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    # Header
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        set_cell_bg(hdr_cells[i], header_color)
        for run in hdr_cells[i].paragraphs[0].runs:
            run.bold = True
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            run.font.size = Pt(10)
    # Rows
    for ridx, row in enumerate(rows):
        row_cells = table.add_row().cells
        for i, val in enumerate(row):
            row_cells[i].text = str(val)
            for run in row_cells[i].paragraphs[0].runs:
                run.font.size = Pt(10)
            if ridx % 2 == 0:
                set_cell_bg(row_cells[i], "EBF3FB")
    return table

# ════════════════════════════════════════════════════════════════════════════
# TITLE PAGE
# ════════════════════════════════════════════════════════════════════════════
title = doc.add_heading("Credit Scoring Model", 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
for run in title.runs:
    run.font.size  = Pt(28)
    run.font.bold  = True
    run.font.color.rgb = RGBColor(0x1a, 0x3a, 0x5c)

sub = doc.add_paragraph("Complete Data Analysis & Machine Learning Report")
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
for run in sub.runs:
    run.font.size   = Pt(16)
    run.font.italic = True
    run.font.color.rgb = RGBColor(0x44, 0x44, 0x44)

doc.add_paragraph()
date_p = doc.add_paragraph(f"Date: {datetime.date.today().strftime('%B %d, %Y')}")
date_p.alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_page_break()

# ════════════════════════════════════════════════════════════════════════════
# 1. EXECUTIVE SUMMARY
# ════════════════════════════════════════════════════════════════════════════
add_heading(doc, "1. Executive Summary", 1)
add_para(doc,
    "This report presents a complete end-to-end credit scoring analysis and "
    "machine learning pipeline. The objective is to predict the creditworthiness "
    "of bank customers — classifying them as Good, Standard, or Poor — based on "
    "their financial history, payment behaviour, and credit profile.")
doc.add_paragraph()
add_para(doc,
    "The analysis covers 100,000 training records and 50,000 test records across "
    "28 financial features. Three classification models were trained and evaluated: "
    "Logistic Regression, Decision Tree, and Random Forest. The Random Forest model "
    "achieved the best performance with 73.3% accuracy and a ROC-AUC of 0.858, "
    "and was used to generate creditworthiness predictions for the full test set.")

doc.add_paragraph()
add_heading(doc, "Key Results", 2)
add_table(doc,
    ["Metric", "Logistic Regression", "Decision Tree", "Random Forest (Best)"],
    [
        ["Accuracy",   "59.90%", "70.91%", "73.33%"],
        ["Precision",  "59.59%", "71.58%", "73.73%"],
        ["Recall",     "59.90%", "70.91%", "73.33%"],
        ["F1-Score",   "58.18%", "71.02%", "73.44%"],
        ["ROC-AUC",    "0.7370", "0.8313", "0.8583"],
        ["CV-Accuracy","60.29%", "69.58%", "70.03%"],
    ]
)

doc.add_page_break()

# ════════════════════════════════════════════════════════════════════════════
# 2. DATASET OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
add_heading(doc, "2. Dataset Overview", 1)
add_para(doc,
    "The dataset contains monthly financial snapshots for 12,500 unique customers "
    "tracked across 8 months, resulting in 100,000 training rows. The test set "
    "contains 50,000 rows for which creditworthiness predictions are generated.")

doc.add_paragraph()
add_heading(doc, "2.1 Dataset Dimensions", 2)
add_table(doc,
    ["Dataset", "Rows", "Columns", "Target"],
    [
        ["train.csv", "100,000", "28", "Credit_Score (Good/Standard/Poor)"],
        ["test.csv",  "50,000",  "27", "Not provided — to be predicted"],
    ]
)

doc.add_paragraph()
add_heading(doc, "2.2 Feature Descriptions", 2)
add_table(doc,
    ["Feature", "Type", "Description"],
    [
        ["Age",                    "Numeric",      "Customer age (contains noisy values)"],
        ["Annual_Income",          "Numeric",      "Gross annual income"],
        ["Monthly_Inhand_Salary",  "Numeric",      "Net monthly take-home salary"],
        ["Num_Bank_Accounts",      "Numeric",      "Number of active bank accounts"],
        ["Num_Credit_Card",        "Numeric",      "Number of credit cards held"],
        ["Interest_Rate",          "Numeric",      "Average interest rate across loans"],
        ["Num_of_Loan",            "Numeric",      "Total number of loans"],
        ["Type_of_Loan",           "Categorical",  "Comma-separated list of loan types"],
        ["Delay_from_due_date",    "Numeric",      "Average days payment was delayed"],
        ["Num_of_Delayed_Payment", "Numeric",      "Count of late/delayed payments"],
        ["Changed_Credit_Limit",   "Numeric",      "Credit limit change amount"],
        ["Outstanding_Debt",       "Numeric",      "Total outstanding debt balance"],
        ["Credit_Utilization_Ratio","Numeric",     "Credit usage as % of limit"],
        ["Credit_History_Age",     "Text",         "Length of credit history (e.g., '5 Years 3 Months')"],
        ["Payment_of_Min_Amount",  "Categorical",  "Whether minimum payment is made (Yes/No)"],
        ["Total_EMI_per_month",    "Numeric",      "Total monthly EMI payment"],
        ["Amount_invested_monthly","Numeric",      "Monthly investment/savings amount"],
        ["Payment_Behaviour",      "Categorical",  "Spending and payment pattern"],
        ["Monthly_Balance",        "Numeric",      "Remaining monthly balance"],
        ["Credit_Mix",             "Categorical",  "Quality of credit portfolio (Good/Standard/Bad)"],
        ["Occupation",             "Categorical",  "Customer occupation"],
        ["Credit_Score",           "Target",       "Creditworthiness label (Good/Standard/Poor)"],
    ]
)

doc.add_page_break()

# ════════════════════════════════════════════════════════════════════════════
# 3. EXPLORATORY DATA ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
add_heading(doc, "3. Exploratory Data Analysis", 1)

add_heading(doc, "3.1 Target Variable Distribution", 2)
add_para(doc,
    "The target variable Credit_Score has three classes. The dataset is "
    "moderately imbalanced, with Standard being the majority class at 53.2%.")

add_table(doc,
    ["Credit Score", "Count", "Proportion"],
    [
        ["Standard", "53,174", "53.17%"],
        ["Poor",     "28,998", "29.00%"],
        ["Good",     "17,828", "17.83%"],
    ]
)
doc.add_paragraph()
add_figure(doc, FIG_DIR / "01_target_distribution.png",
           "Figure 1: Credit Score class distribution and proportions", width=5.8)

doc.add_page_break()
add_heading(doc, "3.2 Missing Values", 2)
add_para(doc,
    "Several columns have missing values. The most significant is "
    "Monthly_Inhand_Salary (15%) and Type_of_Loan (11.4%). "
    "All missing values are handled via median imputation during preprocessing.")

add_table(doc,
    ["Column", "Missing Count", "Missing %"],
    [
        ["Monthly_Inhand_Salary",   "15,002", "15.00%"],
        ["Type_of_Loan",            "11,408", "11.41%"],
        ["Name",                     "9,985",  "9.98%"],
        ["Credit_History_Age",       "9,030",  "9.03%"],
        ["Num_of_Delayed_Payment",   "7,002",  "7.00%"],
        ["Amount_invested_monthly",  "4,479",  "4.48%"],
        ["Num_Credit_Inquiries",     "1,965",  "1.96%"],
        ["Monthly_Balance",          "1,200",  "1.20%"],
    ]
)
doc.add_paragraph()
add_figure(doc, FIG_DIR / "02_missing_values.png",
           "Figure 2: Missing values per column", width=5.5)

doc.add_page_break()
add_heading(doc, "3.3 Numerical Feature Distributions", 2)
add_para(doc,
    "Numerical features show wide variation and significant outliers. "
    "Several features (e.g., Num_Bank_Accounts, Interest_Rate) have extreme "
    "outlier values that were handled through robust scaling and clipping.")
doc.add_paragraph()
add_figure(doc, FIG_DIR / "03_numerical_distributions.png",
           "Figure 3: Distribution of all numerical features", width=6.2)

doc.add_page_break()
add_heading(doc, "3.4 Key Features vs Credit Score", 2)
add_para(doc,
    "Boxplots reveal clear separation between credit score classes for "
    "Outstanding Debt, Interest Rate, and Num_of_Delayed_Payment, "
    "confirming their predictive importance.")
doc.add_paragraph()
add_figure(doc, FIG_DIR / "04_boxplots_by_credit_score.png",
           "Figure 4: Key numerical features by credit score class", width=6.2)

doc.add_page_break()
add_heading(doc, "3.5 Categorical Features vs Credit Score", 2)
add_para(doc,
    "Credit_Mix is the most discriminative categorical feature. Customers "
    "with Good credit mix have ~60% probability of Good credit score, "
    "while Bad credit mix strongly predicts Poor score.")
doc.add_paragraph()
add_figure(doc, FIG_DIR / "05_categorical_vs_credit_score.png",
           "Figure 5: Categorical features vs credit score", width=6.2)

doc.add_page_break()

# ════════════════════════════════════════════════════════════════════════════
# 4. CORRELATION ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
add_heading(doc, "4. Correlation Analysis", 1)

add_heading(doc, "4.1 Full Correlation Heatmap", 2)
add_para(doc,
    "The correlation heatmap shows relationships between all numerical features. "
    "Outstanding Debt and Delay_from_due_date show the strongest negative "
    "correlation with the credit score.")
doc.add_paragraph()
add_figure(doc, FIG_DIR / "06_correlation_heatmap.png",
           "Figure 6: Correlation heatmap of numerical features", width=6.0)

doc.add_paragraph()
add_heading(doc, "4.2 Feature Correlations with Target", 2)
add_figure(doc, FIG_DIR / "07_feature_correlation_with_target.png",
           "Figure 7: Individual feature correlations with credit score", width=5.5)

doc.add_page_break()

# ════════════════════════════════════════════════════════════════════════════
# 5. FINANCIAL ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
add_heading(doc, "5. Financial Feature Analysis", 1)

add_heading(doc, "5.1 Income Distribution", 2)
add_para(doc,
    "Annual income distributions overlap significantly across credit score classes, "
    "suggesting income alone is not a reliable predictor. "
    "However, Good customers show higher median income values.")
add_figure(doc, FIG_DIR / "08_income_by_credit_score.png",
           "Figure 8: Annual income distribution by credit score", width=5.8)

doc.add_paragraph()
add_heading(doc, "5.2 Delayed Payments Analysis", 2)
add_para(doc,
    "Number of delayed payments is one of the strongest discriminators. "
    "Poor credit customers have significantly more delayed payments on average.")
add_figure(doc, FIG_DIR / "09_delayed_payments_by_score.png",
           "Figure 9: Delayed payments distribution by credit score", width=5.8)

doc.add_page_break()
add_heading(doc, "5.3 Outstanding Debt Analysis", 2)
add_para(doc,
    "Outstanding Debt shows the clearest separation: Good customers average "
    "$801 outstanding debt vs $2,081 for Poor customers — a 2.6x difference.")

add_table(doc,
    ["Credit Score", "Mean Outstanding Debt", "Mean Delayed Payments", "Mean Monthly Balance"],
    [
        ["Good",     "$801",   "26", "$401"],
        ["Standard", "$1,278", "32", "$344"],
        ["Poor",     "$2,081", "32", "$299"],
    ]
)
doc.add_paragraph()
add_figure(doc, FIG_DIR / "10_outstanding_debt_by_score.png",
           "Figure 10: Outstanding debt distribution by credit score", width=5.8)

doc.add_paragraph()
add_heading(doc, "5.4 Monthly Balance Distribution", 2)
add_figure(doc, FIG_DIR / "11_monthly_balance_violin.png",
           "Figure 11: Monthly balance violin plot by credit score", width=5.5)

doc.add_page_break()

# ════════════════════════════════════════════════════════════════════════════
# 6. FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════════════════
add_heading(doc, "6. Feature Engineering", 1)
add_para(doc,
    "Seven new features were engineered from existing financial data "
    "to better capture financial health ratios and credit behaviour:")

add_table(doc,
    ["Feature", "Formula", "Business Meaning"],
    [
        ["Debt_to_Income",       "Outstanding Debt / Annual Income",           "Debt burden relative to income"],
        ["EMI_to_Salary",        "Total EMI / Monthly Salary",                 "Loan repayment burden"],
        ["Savings_Rate",         "Amount Invested / Monthly Salary",           "Financial discipline proxy"],
        ["Balance_Buffer",       "Monthly Balance / Monthly Salary",           "Cash cushion relative to income"],
        ["Cards_per_Account",    "Num Credit Cards / Num Bank Accounts",       "Credit card concentration"],
        ["Credit_History_Months","Parsed from 'X Years and Y Months'",         "Credit history length in months"],
        ["Num_Loan_Types",       "Count of loans in Type_of_Loan string",      "Credit diversity"],
    ]
)

doc.add_paragraph()
add_heading(doc, "Preprocessing Pipeline", 2)
add_para(doc, "The following steps were applied to prevent data leakage and ensure clean inputs:")
steps = [
    "1. Drop identifier columns: ID, Customer_ID, Name, SSN, Month",
    "2. Strip special characters from noisy numeric fields",
    "3. Parse Credit_History_Age text into numeric months",
    "4. Count loan types from comma-separated Type_of_Loan string",
    "5. Engineer 5 financial ratio features",
    "6. Label-encode all categorical features (Occupation, Credit_Mix, Payment_Behaviour, Payment_of_Min_Amount)",
    "7. Replace inf/-inf values with NaN, then impute with column median",
    "8. StandardScaler applied to all features (fit only on train set — no leakage)",
    "9. Final NaN/inf safety pass after scaling",
    "10. Stratified 80/20 train/validation split"
]
for step in steps:
    doc.add_paragraph(step, style="List Bullet")

doc.add_page_break()

# ════════════════════════════════════════════════════════════════════════════
# 7. MODEL TRAINING & EVALUATION
# ════════════════════════════════════════════════════════════════════════════
add_heading(doc, "7. Model Training & Evaluation", 1)
add_para(doc,
    "Three classification models were trained and evaluated on the 20,000-record "
    "validation set (stratified split). All models were assessed on Accuracy, "
    "Precision, Recall, F1-Score, ROC-AUC, and 5-fold cross-validated accuracy.")

add_heading(doc, "7.1 Performance Comparison", 2)
add_figure(doc, FIG_DIR / "12_model_comparison.png",
           "Figure 12: Model performance comparison across all metrics", width=6.0)

doc.add_paragraph()
add_heading(doc, "7.2 Confusion Matrices", 2)
add_para(doc,
    "The confusion matrices reveal that all models perform best on the Standard class "
    "(majority class) and have more difficulty with Good (minority class). "
    "Random Forest shows the best balance across all three classes.")
add_figure(doc, FIG_DIR / "13_confusion_matrices.png",
           "Figure 13: Confusion matrices for all three models", width=6.2)

doc.add_page_break()
add_heading(doc, "7.3 ROC Curves", 2)
add_para(doc,
    "ROC curves use one-vs-rest (OvR) strategy for multi-class evaluation. "
    "Random Forest achieves AUC > 0.85 for all classes, demonstrating "
    "strong discriminative ability.")
add_figure(doc, FIG_DIR / "14_roc_curves.png",
           "Figure 14: ROC curves (OvR) for all models", width=6.2)

doc.add_paragraph()
add_heading(doc, "7.4 Feature Importance", 2)
add_para(doc,
    "Random Forest feature importance reveals the top predictors of creditworthiness. "
    "Financial stress indicators (Outstanding Debt, EMI ratio, Delayed Payments) "
    "and credit history are most informative.")
add_figure(doc, FIG_DIR / "15_feature_importance.png",
           "Figure 15: Top 20 feature importances from Random Forest", width=5.8)

doc.add_page_break()

# ════════════════════════════════════════════════════════════════════════════
# 8. FINAL MODEL & PREDICTIONS
# ════════════════════════════════════════════════════════════════════════════
add_heading(doc, "8. Final Model & Test Predictions", 1)

add_heading(doc, "8.1 Best Model: Random Forest", 2)
add_para(doc,
    "Random Forest was selected as the best model based on highest F1-Score (0.7344) "
    "and ROC-AUC (0.8583). The model was retrained on the full 100,000-record "
    "training set before generating test predictions.")

add_table(doc,
    ["Parameter", "Value"],
    [
        ["Algorithm",    "Random Forest Classifier"],
        ["n_estimators", "200"],
        ["max_depth",    "12"],
        ["random_state", "42"],
        ["Accuracy",     "73.33%"],
        ["F1-Score",     "73.44%"],
        ["ROC-AUC",      "0.8583"],
    ]
)

doc.add_paragraph()
add_heading(doc, "8.2 Test Prediction Distribution", 2)
add_para(doc,
    "Predictions were generated for all 50,000 test records, "
    "with per-class probabilities included for each customer.")

add_table(doc,
    ["Predicted Score", "Count", "Proportion"],
    [
        ["Standard", "26,022", "52.0%"],
        ["Poor",     "13,534", "27.1%"],
        ["Good",     "10,444", "20.9%"],
    ]
)
doc.add_paragraph()
add_figure(doc, FIG_DIR / "16_test_predictions_distribution.png",
           "Figure 16: Predicted credit score distribution on test set", width=5.0)

doc.add_page_break()

# ════════════════════════════════════════════════════════════════════════════
# 9. KEY INSIGHTS
# ════════════════════════════════════════════════════════════════════════════
add_heading(doc, "9. Key Insights & Conclusions", 1)

insights = [
    ("Outstanding Debt is the #1 predictor",
     "Good customers average $801 outstanding debt vs $2,081 for Poor — a 2.6x difference. "
     "Debt-to-Income ratio (engineered feature) is among the top 5 most important features."),

    ("Credit History Age matters significantly",
     "Longer credit history strongly correlates with Good credit scores. "
     "Customers with 20+ years of credit history are overwhelmingly Good or Standard."),

    ("Payment discipline is critical",
     "The number of delayed payments and days delayed are strong risk indicators. "
     "Good customers average 26 delayed payments vs 32 for Poor customers."),

    ("Credit Mix quality is highly discriminative",
     "Customers with a 'Good' credit mix have ~60% probability of Good credit score. "
     "This is the most informative categorical feature."),

    ("Income alone is not sufficient",
     "Annual income distributions overlap significantly between classes. "
     "High income does not guarantee Good credit — debt management is more important."),

    ("Engineered features add significant value",
     "EMI-to-Salary ratio and Debt-to-Income ratio are among the top 10 most "
     "important features, validating the feature engineering step."),

    ("Data quality issues are prevalent",
     "The raw dataset contains significant noise: negative ages, special characters, "
     "out-of-range values (e.g., Interest Rate = 5797), and inconsistent formats. "
     "Robust preprocessing is essential."),
]

for title_text, body_text in insights:
    add_para(doc, f"• {title_text}", bold=True)
    add_para(doc, f"  {body_text}")
    doc.add_paragraph()

doc.add_page_break()

# ════════════════════════════════════════════════════════════════════════════
# 10. OUTPUT FILES
# ════════════════════════════════════════════════════════════════════════════
add_heading(doc, "10. Project Deliverables", 1)

add_table(doc,
    ["File", "Location", "Description"],
    [
        ["analysis.py",               "scripts/",                  "Full EDA script"],
        ["model.py",                  "scripts/",                  "Feature engineering + ML training"],
        ["credit_scoring_analysis.ipynb", "notebooks/",            "Complete Jupyter notebook"],
        ["train.csv",                 "data/",                     "Training dataset (100K records)"],
        ["test.csv",                  "data/",                     "Test dataset (50K records)"],
        ["test_predictions.csv",      "outputs/predictions/",      "50,000 test predictions with probabilities"],
        ["model_comparison.csv",      "outputs/predictions/",      "Model metrics comparison table"],
        ["best_model.pkl",            "outputs/models/",           "Saved Random Forest model"],
        ["model_metadata.json",       "outputs/models/",           "Model configuration and metrics"],
        ["Figures 01-16",             "outputs/figures/",          "16 visualisation charts"],
        ["credit_scoring_report.docx","reports/",                  "This Word report"],
        ["README.md",                 "./",                        "Project documentation"],
        ["requirements.txt",          "./",                        "Python dependencies"],
    ]
)

doc.add_paragraph()
add_heading(doc, "Project Structure", 2)
structure = """Credit Scoring Model/
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   └── credit_scoring_analysis.ipynb
├── scripts/
│   ├── analysis.py
│   └── model.py
├── outputs/
│   ├── figures/          (16 charts)
│   ├── models/           (best_model.pkl, model_metadata.json)
│   └── predictions/      (test_predictions.csv, model_comparison.csv)
├── reports/
│   └── credit_scoring_report.docx
├── README.md
└── requirements.txt"""
doc.add_paragraph(structure, style="No Spacing")
for para in doc.paragraphs[-1:]:
    for run in para.runs:
        run.font.name = "Courier New"
        run.font.size = Pt(9)

# ════════════════════════════════════════════════════════════════════════════
# Save
# ════════════════════════════════════════════════════════════════════════════
out_path = REPORTS_DIR / "credit_scoring_report.docx"
doc.save(str(out_path))
print(f"Report saved → {out_path}")

"""
Credit Scoring Model - Feature Engineering + ML Training
=========================================================
Preprocessing, feature engineering, model building,
evaluation, ROC curves, confusion matrices, feature
importance, final model selection and test predictions.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
FIG_DIR    = Path("outputs/figures")
MODEL_DIR  = Path("outputs/models")
PRED_DIR   = Path("outputs/predictions")
for d in [FIG_DIR, MODEL_DIR, PRED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)

TARGET     = "Credit_Score"
LABEL_MAP  = {"Poor": 0, "Standard": 1, "Good": 2}
INV_LABEL  = {v: k for k, v in LABEL_MAP.items()}

# ── 1. Load data ────────────────────────────────────────────────────────────
print("=" * 65)
print("  CREDIT SCORING — FEATURE ENGINEERING + ML MODELING")
print("=" * 65)

train_raw = pd.read_csv(DATA_DIR / "train.csv", low_memory=False)
test_raw  = pd.read_csv(DATA_DIR / "test.csv",  low_memory=False)
print(f"\n[1] Loaded train {train_raw.shape}, test {test_raw.shape}")

# ── 2. Preprocessing pipeline ───────────────────────────────────────────────

DROP_COLS = ["ID", "Customer_ID", "Name", "SSN", "Month"]

def clean_numeric(series):
    """Strip non-numeric chars and coerce to float."""
    return pd.to_numeric(
        series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
        errors="coerce"
    )

def parse_credit_history_age(series):
    """Convert 'X Years and Y Months' to total months."""
    years  = series.astype(str).str.extract(r"(\d+)\s*Year",  expand=False)
    months = series.astype(str).str.extract(r"(\d+)\s*Month", expand=False)
    y = pd.to_numeric(years,  errors="coerce").fillna(0)
    m = pd.to_numeric(months, errors="coerce").fillna(0)
    result = y * 12 + m
    result[result == 0] = np.nan
    return result

NUM_COLS = [
    "Age", "Annual_Income", "Monthly_Inhand_Salary",
    "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate",
    "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
    "Changed_Credit_Limit", "Num_Credit_Inquiries",
    "Outstanding_Debt", "Credit_Utilization_Ratio",
    "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance",
]

CAT_COLS = ["Occupation", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"]

def preprocess(df, is_train=True, scaler=None, cat_encoders=None, fit=True):
    df = df.copy()
    # Drop leaky / identifier columns
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    # Numeric cleaning
    for col in NUM_COLS:
        if col in df.columns:
            df[col] = clean_numeric(df[col])

    # Credit history age
    if "Credit_History_Age" in df.columns:
        df["Credit_History_Months"] = parse_credit_history_age(df["Credit_History_Age"])
        df.drop(columns=["Credit_History_Age"], inplace=True)

    # Type_of_Loan — count number of loans in the string
    if "Type_of_Loan" in df.columns:
        df["Num_Loan_Types"] = df["Type_of_Loan"].fillna("").astype(str).apply(
            lambda x: len([t for t in x.split(",") if t.strip() not in ("", "nan", "none", "")])
        )
        df.drop(columns=["Type_of_Loan"], inplace=True)

    # ── Feature Engineering ─────────────────────────────────────────────
    eps = 1e-6
    # Debt-to-income ratio
    if "Outstanding_Debt" in df.columns and "Annual_Income" in df.columns:
        df["Debt_to_Income"] = df["Outstanding_Debt"] / (df["Annual_Income"] + eps)

    # EMI burden
    if "Total_EMI_per_month" in df.columns and "Monthly_Inhand_Salary" in df.columns:
        df["EMI_to_Salary"] = df["Total_EMI_per_month"] / (df["Monthly_Inhand_Salary"] + eps)

    # Savings rate
    if "Amount_invested_monthly" in df.columns and "Monthly_Inhand_Salary" in df.columns:
        df["Savings_Rate"] = df["Amount_invested_monthly"] / (df["Monthly_Inhand_Salary"] + eps)

    # Balance buffer
    if "Monthly_Balance" in df.columns and "Monthly_Inhand_Salary" in df.columns:
        df["Balance_Buffer"] = df["Monthly_Balance"] / (df["Monthly_Inhand_Salary"] + eps)

    # Credit card utilization proxy
    if "Num_Credit_Card" in df.columns and "Num_Bank_Accounts" in df.columns:
        df["Cards_per_Account"] = df["Num_Credit_Card"] / (df["Num_Bank_Accounts"] + eps)

    # ── Encode categoricals ─────────────────────────────────────────────
    if cat_encoders is None:
        cat_encoders = {}

    for col in CAT_COLS:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan, "_": np.nan})
        df[col] = df[col].fillna("Unknown")
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            cat_encoders[col] = le
        else:
            le = cat_encoders[col]
            # handle unseen
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
            df[col] = le.transform(df[col])

    # ── Drop remaining non-numeric / object columns ─────────────────────
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    if is_train and TARGET in obj_cols:
        obj_cols.remove(TARGET)
    df.drop(columns=obj_cols, inplace=True)

    # ── Replace inf / -inf with NaN ──────────────────────────────────────
    feature_cols = [c for c in df.columns if c != TARGET]
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    # ── Fill remaining NaN with median ───────────────────────────────────
    for col in feature_cols:
        if df[col].isnull().any():
            med = df[col].median()
            df[col] = df[col].fillna(0 if pd.isna(med) else med)

    # ── Scale ────────────────────────────────────────────────────────────
    feature_cols = [c for c in df.columns if c != TARGET]
    if fit:
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    else:
        df[feature_cols] = scaler.transform(df[feature_cols])

    # ── Final safety: replace any residual NaN/inf after scaling ─────────
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)

    return df, scaler, cat_encoders

print("\n[2] Preprocessing train set …")
train_proc, scaler, cat_encoders = preprocess(train_raw, is_train=True, fit=True)
print(f"    Processed shape : {train_proc.shape}")

print("[3] Preprocessing test set …")
test_proc, _, _ = preprocess(test_raw, is_train=False, scaler=scaler,
                              cat_encoders=cat_encoders, fit=False)
print(f"    Processed shape : {test_proc.shape}")

# ── 3. Prepare X / y ────────────────────────────────────────────────────────
X = train_proc.drop(columns=[TARGET])
y = train_proc[TARGET].map(LABEL_MAP)

print(f"\n[4] Features used ({len(X.columns)}): {list(X.columns)}")
print(f"    Class distribution:\n{y.value_counts().rename(INV_LABEL).to_string()}")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n[5] Train split: {X_train.shape}, Val split: {X_val.shape}")

# ── 4. Train models ──────────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, C=0.5),
    "Decision Tree":       DecisionTreeClassifier(max_depth=8, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=12,
                                                   random_state=42, n_jobs=-1),
}

results = {}
print("\n[6] Training models …")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_val)
    y_proba = model.predict_proba(X_val)

    acc  = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_val, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_val, y_pred, average="weighted", zero_division=0)
    auc  = roc_auc_score(y_val, y_proba, multi_class="ovr", average="weighted")
    cv   = cross_val_score(model, X, y, cv=StratifiedKFold(5), scoring="accuracy").mean()

    results[name] = {
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(auc, 4),
        "CV-Accuracy": round(cv, 4),
        "model": model,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }
    print(f"\n  -- {name} --")
    print(f"     Accuracy  : {acc:.4f}")
    print(f"     Precision : {prec:.4f}")
    print(f"     Recall    : {rec:.4f}")
    print(f"     F1-Score  : {f1:.4f}")
    print(f"     ROC-AUC   : {auc:.4f}")
    print(f"     CV-Acc    : {cv:.4f}")
    print(classification_report(y_val, y_pred,
          target_names=[INV_LABEL[i] for i in sorted(INV_LABEL)]))

# ── 5. Comparison table ──────────────────────────────────────────────────────
metrics_df = pd.DataFrame({
    name: {k: v for k, v in info.items() if k not in ("model", "y_pred", "y_proba")}
    for name, info in results.items()
}).T

print("\n[7] Model Comparison")
print(metrics_df.to_string())
metrics_df.to_csv(PRED_DIR / "model_comparison.csv")

# ── Fig 12 : Model comparison bar chart ─────────────────────────────────────
metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
x = np.arange(len(metric_names))
width = 0.25
colors_bar = ["#3498db", "#e67e22", "#2ecc71"]

fig, ax = plt.subplots(figsize=(13, 6))
for i, (name, info) in enumerate(results.items()):
    vals = [info[m] for m in metric_names]
    bars = ax.bar(x + i * width, vals, width, label=name,
                  color=colors_bar[i], edgecolor="white", alpha=0.88)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7.5)

ax.set_xticks(x + width)
ax.set_xticklabels(metric_names)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score")
ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "12_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("[FIG] Saved 12_model_comparison.png")

# ── Fig 13 : Confusion matrices ──────────────────────────────────────────────
class_names = [INV_LABEL[i] for i in sorted(INV_LABEL)]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, info) in zip(axes, results.items()):
    cm = confusion_matrix(y_val, info["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f"{name}\nConfusion Matrix", fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(FIG_DIR / "13_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("[FIG] Saved 13_confusion_matrices.png")

# ── Fig 14 : ROC curves (OvR) ────────────────────────────────────────────────
from sklearn.preprocessing import label_binarize
y_val_bin = label_binarize(y_val, classes=[0, 1, 2])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, info) in zip(axes, results.items()):
    y_proba = info["y_proba"]
    line_styles = ["-", "--", "-."]
    line_colors = ["#e74c3c", "#3498db", "#2ecc71"]
    for i, (cls, ls, lc) in enumerate(zip(class_names, line_styles, line_colors)):
        fpr, tpr, _ = roc_curve(y_val_bin[:, i], y_proba[:, i])
        auc_cls = roc_auc_score(y_val_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, linestyle=ls, color=lc,
                label=f"{cls} (AUC={auc_cls:.2f})", linewidth=1.8)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_title(f"{name}\nROC Curve", fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "14_roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("[FIG] Saved 14_roc_curves.png")

# ── Fig 15 : Feature importance (RF) ────────────────────────────────────────
rf_model = results["Random Forest"]["model"]
fi = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))
fi.head(20).sort_values().plot(kind="barh", ax=ax, color="#8e44ad", edgecolor="white")
ax.set_title("Top 20 Feature Importances (Random Forest)", fontsize=13, fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig(FIG_DIR / "15_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("[FIG] Saved 15_feature_importance.png")

# ── 6. Select best model ──────────────────────────────────────────────────────
best_name = metrics_df["F1-Score"].astype(float).idxmax()
best_model = results[best_name]["model"]
print(f"\n[8] Best model: {best_name} (F1={metrics_df.loc[best_name,'F1-Score']})")

# Retrain on full data
best_model.fit(X, y)
print(f"    Retrained {best_name} on full training set.")

# Save model
joblib.dump({"model": best_model, "scaler": scaler,
             "cat_encoders": cat_encoders, "features": list(X.columns)},
            MODEL_DIR / "best_model.pkl")
print(f"    Model saved → outputs/models/best_model.pkl")

# Save metadata
with open(MODEL_DIR / "model_metadata.json", "w") as f:
    json.dump({
        "best_model": best_name,
        "features": list(X.columns),
        "label_map": LABEL_MAP,
        "metrics": {k: {m: v for m, v in info.items()
                         if m not in ("model", "y_pred", "y_proba")}
                    for k, info in results.items()}
    }, f, indent=2)

# ── 7. Generate test predictions ─────────────────────────────────────────────
X_test = test_proc[[c for c in X.columns if c in test_proc.columns]]
# fill any missing feature columns with 0
for col in X.columns:
    if col not in X_test.columns:
        X_test[col] = 0
X_test = X_test[X.columns]

# fill NaN
for col in X_test.columns:
    if X_test[col].isnull().any():
        X_test[col].fillna(0, inplace=True)

y_test_pred  = best_model.predict(X_test)
y_test_proba = best_model.predict_proba(X_test)

pred_labels = [INV_LABEL[p] for p in y_test_pred]

pred_df = test_raw[["ID", "Customer_ID"]].copy()
pred_df["Predicted_Credit_Score"] = pred_labels
pred_df["Prob_Poor"]     = y_test_proba[:, 0].round(4)
pred_df["Prob_Standard"] = y_test_proba[:, 1].round(4)
pred_df["Prob_Good"]     = y_test_proba[:, 2].round(4)

pred_df.to_csv(PRED_DIR / "test_predictions.csv", index=False)
print(f"\n[9] Test predictions saved → outputs/predictions/test_predictions.csv")
print(f"    Prediction distribution:\n{pred_df['Predicted_Credit_Score'].value_counts().to_string()}")

# ── Fig 16 : Predicted score distribution ───────────────────────────────────
pal = {"Good": "#2ecc71", "Standard": "#f39c12", "Poor": "#e74c3c"}
fig, ax = plt.subplots(figsize=(8, 5))
vc = pred_df["Predicted_Credit_Score"].value_counts()
colors = [pal.get(k, "#95a5a6") for k in vc.index]
ax.bar(vc.index, vc.values, color=colors, edgecolor="white", linewidth=1.2)
for i, (k, v) in enumerate(zip(vc.index, vc.values)):
    ax.text(i, v + 0.5, str(v), ha="center", fontsize=10)
ax.set_title("Predicted Credit Score Distribution (Test Set)", fontweight="bold")
ax.set_xlabel("Predicted Credit Score"); ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(FIG_DIR / "16_test_predictions_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("[FIG] Saved 16_test_predictions_distribution.png")

print("\n" + "=" * 65)
print("  MODELING COMPLETE")
print(f"  Best Model  : {best_name}")
print(f"  Accuracy    : {results[best_name]['Accuracy']:.4f}")
print(f"  F1-Score    : {results[best_name]['F1-Score']:.4f}")
print(f"  ROC-AUC     : {results[best_name]['ROC-AUC']:.4f}")
print("=" * 65)

"""
Credit Scoring Model - Exploratory Data Analysis
=================================================
Full EDA: dataset inspection, statistics, missing values,
distributions, correlations, and key insights.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
FIG_DIR    = Path("outputs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {"Good": "#2ecc71", "Standard": "#f39c12", "Poor": "#e74c3c"}
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# ── 1. Load data ────────────────────────────────────────────────────────────
print("=" * 65)
print("  CREDIT SCORING — EXPLORATORY DATA ANALYSIS")
print("=" * 65)

train = pd.read_csv(DATA_DIR / "train.csv", low_memory=False)
test  = pd.read_csv(DATA_DIR / "test.csv",  low_memory=False)

print(f"\n[1] Dataset Shapes")
print(f"    Train : {train.shape[0]:,} rows × {train.shape[1]} columns")
print(f"    Test  : {test.shape[0]:,} rows × {test.shape[1]} columns")

# ── 2. Column overview ──────────────────────────────────────────────────────
print(f"\n[2] Columns ({train.shape[1]})")
print("   ", list(train.columns))

# ── 3. Data types ───────────────────────────────────────────────────────────
print(f"\n[3] Data types")
print(train.dtypes.to_string())

# ── 4. Basic statistics ─────────────────────────────────────────────────────
print(f"\n[4] Basic statistics (numeric columns)")
print(train.describe(include="all").T.to_string())

# ── 5. Missing values ───────────────────────────────────────────────────────
miss_train = train.isnull().sum()
miss_pct   = (miss_train / len(train) * 100).round(2)
missing_df = pd.DataFrame({"Missing": miss_train, "Pct%": miss_pct})
missing_df = missing_df[missing_df["Missing"] > 0].sort_values("Missing", ascending=False)
print(f"\n[5] Missing values in train")
print(missing_df.to_string() if not missing_df.empty else "    None")

# ── 6. Duplicate records ────────────────────────────────────────────────────
dups = train.duplicated().sum()
print(f"\n[6] Duplicate rows in train: {dups}")

# ── 7. Target variable ──────────────────────────────────────────────────────
target_col = "Credit_Score"
print(f"\n[7] Target variable : '{target_col}'")
print(train[target_col].value_counts().to_string())
print(f"    Proportions:")
print((train[target_col].value_counts(normalize=True) * 100).round(2).to_string())

# ────────────────────────────────────────────────────────────────────────────
# VISUALIZATIONS
# ────────────────────────────────────────────────────────────────────────────

# ── Fig 1 : Target distribution ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
vc = train[target_col].value_counts()
colors = [PALETTE.get(k, "#95a5a6") for k in vc.index]

axes[0].bar(vc.index, vc.values, color=colors, edgecolor="white", linewidth=1.2)
axes[0].set_title("Credit Score Distribution", fontweight="bold")
axes[0].set_xlabel("Credit Score"); axes[0].set_ylabel("Count")
for i, (k, v) in enumerate(zip(vc.index, vc.values)):
    axes[0].text(i, v + 50, f"{v:,}", ha="center", fontsize=10)

axes[1].pie(vc.values, labels=vc.index, colors=colors, autopct="%1.1f%%",
            startangle=140, wedgeprops=dict(edgecolor="white"))
axes[1].set_title("Credit Score Proportion", fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "01_target_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[FIG] Saved 01_target_distribution.png")

# ── Fig 2 : Missing value heatmap ───────────────────────────────────────────
if not missing_df.empty:
    fig, ax = plt.subplots(figsize=(10, 5))
    missing_df["Missing"].sort_values().plot(kind="barh", ax=ax, color="#e67e22")
    ax.set_title("Missing Values per Column", fontweight="bold")
    ax.set_xlabel("Count")
    for i, v in enumerate(missing_df["Missing"].sort_values()):
        ax.text(v + 10, i, str(v), va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_missing_values.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[FIG] Saved 02_missing_values.png")

# ── 8. Clean / cast key numeric columns ────────────────────────────────────
def clean_numeric(series):
    return pd.to_numeric(series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
                         errors="coerce")

num_cols_raw = ["Age", "Annual_Income", "Monthly_Inhand_Salary",
                "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate",
                "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
                "Num_Credit_Inquiries", "Outstanding_Debt",
                "Credit_Utilization_Ratio", "Total_EMI_per_month",
                "Amount_invested_monthly", "Monthly_Balance",
                "Changed_Credit_Limit"]

for col in num_cols_raw:
    if col in train.columns:
        train[col] = clean_numeric(train[col])

print("\n[8] Numeric casting complete")

# ── Fig 3 : Numerical distributions ─────────────────────────────────────────
plot_cols = [c for c in num_cols_raw if c in train.columns]
n = len(plot_cols)
ncols = 4
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.5))
axes = axes.flatten()
for i, col in enumerate(plot_cols):
    data = train[col].dropna()
    axes[i].hist(data, bins=40, color="#3498db", edgecolor="white", alpha=0.85)
    axes[i].set_title(col, fontsize=9, fontweight="bold")
    axes[i].set_xlabel(""); axes[i].set_ylabel("Freq")
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Numerical Feature Distributions", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(FIG_DIR / "03_numerical_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("[FIG] Saved 03_numerical_distributions.png")

# ── Fig 4 : Boxplots by Credit Score ─────────────────────────────────────────
key_num = ["Annual_Income", "Monthly_Inhand_Salary", "Outstanding_Debt",
           "Credit_Utilization_Ratio", "Num_of_Delayed_Payment",
           "Total_EMI_per_month", "Monthly_Balance", "Interest_Rate"]
key_num = [c for c in key_num if c in train.columns]

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()
order = ["Good", "Standard", "Poor"]
for i, col in enumerate(key_num):
    data_list = [train.loc[train[target_col] == cat, col].dropna() for cat in order]
    bp = axes[i].boxplot(data_list, patch_artist=True,
                         medianprops=dict(color="black", linewidth=2))
    for patch, cat in zip(bp["boxes"], order):
        patch.set_facecolor(PALETTE.get(cat, "#95a5a6"))
        patch.set_alpha(0.8)
    axes[i].set_xticklabels(order, fontsize=9)
    axes[i].set_title(col, fontsize=9, fontweight="bold")
plt.suptitle("Key Features vs Credit Score", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "04_boxplots_by_credit_score.png", dpi=150, bbox_inches="tight")
plt.close()
print("[FIG] Saved 04_boxplots_by_credit_score.png")

# ── Fig 5 : Categorical features ─────────────────────────────────────────────
cat_cols = ["Occupation", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"]
cat_cols = [c for c in cat_cols if c in train.columns]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()
for i, col in enumerate(cat_cols):
    ct = pd.crosstab(train[col], train[target_col], normalize="index") * 100
    ct = ct[[c for c in ["Good", "Standard", "Poor"] if c in ct.columns]]
    colors = [PALETTE.get(c, "#95a5a6") for c in ct.columns]
    ct.plot(kind="bar", ax=axes[i], color=colors, edgecolor="white", width=0.7)
    axes[i].set_title(f"{col} vs Credit Score (%)", fontweight="bold")
    axes[i].set_xlabel(""); axes[i].set_ylabel("% within category")
    axes[i].tick_params(axis="x", rotation=35)
    axes[i].legend(title="Credit Score", fontsize=8)
plt.suptitle("Categorical Features vs Credit Score", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "05_categorical_vs_credit_score.png", dpi=150, bbox_inches="tight")
plt.close()
print("[FIG] Saved 05_categorical_vs_credit_score.png")

# ── Fig 6 : Correlation heatmap ──────────────────────────────────────────────
label_map = {"Good": 2, "Standard": 1, "Poor": 0}
train["Credit_Score_Num"] = train[target_col].map(label_map)
corr_cols = [c for c in num_cols_raw if c in train.columns] + ["Credit_Score_Num"]
corr_mat = train[corr_cols].corr()

fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr_mat, dtype=bool))
sns.heatmap(corr_mat, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, linewidths=0.4, ax=ax, annot_kws={"size": 7})
ax.set_title("Correlation Heatmap (Numerical Features)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "06_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("[FIG] Saved 06_correlation_heatmap.png")

# ── Fig 7 : Top correlations with target ────────────────────────────────────
target_corr = corr_mat["Credit_Score_Num"].drop("Credit_Score_Num").sort_values()
colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in target_corr.values]

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(target_corr.index, target_corr.values, color=colors, edgecolor="white")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Feature Correlation with Credit Score", fontsize=13, fontweight="bold")
ax.set_xlabel("Pearson Correlation")
plt.tight_layout()
plt.savefig(FIG_DIR / "07_feature_correlation_with_target.png", dpi=150, bbox_inches="tight")
plt.close()
print("[FIG] Saved 07_feature_correlation_with_target.png")

# ── Fig 8 : Income distribution by score ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for score, color in PALETTE.items():
    subset = train.loc[train[target_col] == score, "Annual_Income"].dropna()
    subset_clipped = subset.clip(upper=subset.quantile(0.99))
    ax.hist(subset_clipped, bins=50, alpha=0.6, color=color, label=score, edgecolor="none")
ax.set_title("Annual Income Distribution by Credit Score", fontweight="bold")
ax.set_xlabel("Annual Income"); ax.set_ylabel("Count")
ax.legend(title="Credit Score")
plt.tight_layout()
plt.savefig(FIG_DIR / "08_income_by_credit_score.png", dpi=150, bbox_inches="tight")
plt.close()
print("[FIG] Saved 08_income_by_credit_score.png")

# ── Fig 9 : Delayed payments vs score ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for score, color in PALETTE.items():
    subset = train.loc[train[target_col] == score, "Num_of_Delayed_Payment"].dropna()
    ax.hist(subset, bins=30, alpha=0.65, color=color, label=score, edgecolor="none")
ax.set_title("Delayed Payments Distribution by Credit Score", fontweight="bold")
ax.set_xlabel("Num of Delayed Payments"); ax.set_ylabel("Count")
ax.legend(title="Credit Score")
plt.tight_layout()
plt.savefig(FIG_DIR / "09_delayed_payments_by_score.png", dpi=150, bbox_inches="tight")
plt.close()
print("[FIG] Saved 09_delayed_payments_by_score.png")

# ── Fig 10 : Outstanding debt vs score ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for score, color in PALETTE.items():
    subset = train.loc[train[target_col] == score, "Outstanding_Debt"].dropna()
    ax.hist(subset, bins=40, alpha=0.65, color=color, label=score, edgecolor="none")
ax.set_title("Outstanding Debt Distribution by Credit Score", fontweight="bold")
ax.set_xlabel("Outstanding Debt ($)"); ax.set_ylabel("Count")
ax.legend(title="Credit Score")
plt.tight_layout()
plt.savefig(FIG_DIR / "10_outstanding_debt_by_score.png", dpi=150, bbox_inches="tight")
plt.close()
print("[FIG] Saved 10_outstanding_debt_by_score.png")

# ── Fig 11 : Monthly balance vs score ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
order = ["Good", "Standard", "Poor"]
data_list = [train.loc[train[target_col] == cat, "Monthly_Balance"].dropna()
             for cat in order]
vp = ax.violinplot(data_list, positions=range(len(order)), showmedians=True)
for i, (body, cat) in enumerate(zip(vp["bodies"], order)):
    body.set_facecolor(PALETTE.get(cat, "#95a5a6"))
    body.set_alpha(0.7)
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order)
ax.set_title("Monthly Balance Distribution by Credit Score", fontweight="bold")
ax.set_ylabel("Monthly Balance ($)")
plt.tight_layout()
plt.savefig(FIG_DIR / "11_monthly_balance_violin.png", dpi=150, bbox_inches="tight")
plt.close()
print("[FIG] Saved 11_monthly_balance_violin.png")

# ── Statistical summaries by target ─────────────────────────────────────────
print("\n[9] Statistical summary by Credit Score")
summary_by_score = train.groupby(target_col)[key_num].agg(["mean", "median", "std"]).round(2)
print(summary_by_score.to_string())

# ── Key insights ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  KEY EDA INSIGHTS")
print("=" * 65)
vc_pct = train[target_col].value_counts(normalize=True) * 100
for k, v in vc_pct.items():
    print(f"  {k:10s}: {v:.1f}%")

for col in ["Annual_Income", "Outstanding_Debt", "Num_of_Delayed_Payment"]:
    if col in train.columns:
        means = train.groupby(target_col)[col].mean().round(2)
        print(f"\n  Mean {col}:")
        for cat, val in means.items():
            print(f"    {cat:10s}: {val:,.2f}")

print("\n[EDA] Analysis complete — all figures saved to outputs/figures/")

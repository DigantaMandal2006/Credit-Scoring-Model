"""
Credit Scoring — Flask REST API
================================
Endpoints:
  GET  /health                  → service health check
  GET  /model/info              → model metadata & performance metrics
  POST /predict                 → predict credit score for one customer
  POST /predict/batch           → predict for a list of customers
  GET  /features                → list expected input features
  GET  /metrics                 → model comparison metrics table
"""

import sys
import os
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

# ── Paths — work from any cwd ────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
MODEL_PKL = BASE_DIR / "outputs" / "models" / "best_model.pkl"
META_JSON = BASE_DIR / "outputs" / "models" / "model_metadata.json"

# Make sure backend/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessor import preprocess_single, preprocess_dataframe, FEATURE_ORDER

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Load model ───────────────────────────────────────────────────────────────
log.info("Loading model from %s", MODEL_PKL)
_bundle      = joblib.load(MODEL_PKL)
MODEL        = _bundle["model"]
SCALER       = _bundle["scaler"]
CAT_ENCODERS = _bundle["cat_encoders"]

with open(META_JSON) as f:
    METADATA = json.load(f)

INV_LABEL = {v: k for k, v in METADATA["label_map"].items()}
log.info("Model loaded — best model: %s", METADATA["best_model"])

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # allow Streamlit (different port) to call the API

# ── Helpers ───────────────────────────────────────────────────────────────────
SCORE_COLORS = {"Good": "#2ecc71", "Standard": "#f39c12", "Poor": "#e74c3c"}
SCORE_EMOJI  = {"Good": "✅", "Standard": "⚠️", "Poor": "❌"}

def _make_prediction(record: dict) -> dict:
    """Run preprocessing + model inference on a single record dict."""
    X = preprocess_single(record, SCALER, CAT_ENCODERS)
    pred_idx  = int(MODEL.predict(X)[0])
    proba     = MODEL.predict_proba(X)[0]
    label     = INV_LABEL[pred_idx]
    return {
        "credit_score":  label,
        "confidence":    round(float(proba[pred_idx]), 4),
        "probabilities": {
            "Poor":     round(float(proba[0]), 4),
            "Standard": round(float(proba[1]), 4),
            "Good":     round(float(proba[2]), 4),
        },
        "color":  SCORE_COLORS[label],
        "emoji":  SCORE_EMOJI[label],
    }


# ════════════════════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":    "ok",
        "model":     METADATA["best_model"],
        "version":   "1.0.0",
        "endpoints": ["/health", "/model/info", "/predict",
                      "/predict/batch", "/features", "/metrics"],
    })


@app.route("/model/info", methods=["GET"])
def model_info():
    return jsonify({
        "best_model":    METADATA["best_model"],
        "num_features":  len(FEATURE_ORDER),
        "feature_names": FEATURE_ORDER,
        "label_map":     METADATA["label_map"],
        "metrics":       METADATA["metrics"],
    })


@app.route("/features", methods=["GET"])
def features():
    """Return the expected raw input features with descriptions."""
    fields = {
        "Age":                    {"type": "number", "description": "Customer age in years", "example": 34},
        "Annual_Income":          {"type": "number", "description": "Annual gross income ($)", "example": 50000},
        "Monthly_Inhand_Salary":  {"type": "number", "description": "Net monthly take-home salary ($)", "example": 3500},
        "Num_Bank_Accounts":      {"type": "integer","description": "Number of active bank accounts", "example": 3},
        "Num_Credit_Card":        {"type": "integer","description": "Number of credit cards", "example": 4},
        "Interest_Rate":          {"type": "number", "description": "Average interest rate (%)", "example": 12},
        "Num_of_Loan":            {"type": "integer","description": "Number of active loans", "example": 2},
        "Delay_from_due_date":    {"type": "integer","description": "Average days payment is delayed", "example": 5},
        "Num_of_Delayed_Payment": {"type": "integer","description": "Count of delayed payments", "example": 3},
        "Changed_Credit_Limit":   {"type": "number", "description": "Credit limit change amount", "example": 500},
        "Num_Credit_Inquiries":   {"type": "integer","description": "Number of credit inquiries", "example": 4},
        "Outstanding_Debt":       {"type": "number", "description": "Total outstanding debt ($)", "example": 1200},
        "Credit_Utilization_Ratio":{"type": "number","description": "Credit usage as % of limit", "example": 28.5},
        "Credit_History_Age":     {"type": "string", "description": "'X Years and Y Months'", "example": "5 Years and 3 Months"},
        "Total_EMI_per_month":    {"type": "number", "description": "Total monthly EMI payments ($)", "example": 200},
        "Amount_invested_monthly":{"type": "number", "description": "Monthly investment amount ($)", "example": 150},
        "Monthly_Balance":        {"type": "number", "description": "Remaining monthly balance ($)", "example": 350},
        "Num_Loan_Types":         {"type": "integer","description": "Number of distinct loan types", "example": 2},
        "Occupation":             {"type": "string", "description": "Customer occupation", "example": "Engineer",
                                   "options": ["Accountant","Architect","Developer","Doctor","Engineer","Entrepreneur",
                                               "Journalist","Lawyer","Manager","Mechanic","Media_Manager","Musician",
                                               "Scientist","Teacher","Writer"]},
        "Credit_Mix":             {"type": "string", "description": "Credit portfolio quality",
                                   "options": ["Good","Standard","Bad"]},
        "Payment_of_Min_Amount":  {"type": "string", "description": "Pays minimum credit card amount?",
                                   "options": ["Yes","No","NM"]},
        "Payment_Behaviour":      {"type": "string", "description": "Spending and payment pattern",
                                   "options": ["High_spent_Large_value_payments",
                                               "High_spent_Medium_value_payments",
                                               "High_spent_Small_value_payments",
                                               "Low_spent_Large_value_payments",
                                               "Low_spent_Medium_value_payments",
                                               "Low_spent_Small_value_payments"]},
        "Type_of_Loan":           {"type": "string", "description": "Comma-separated loan types",
                                   "example": "Auto Loan, Personal Loan"},
    }
    return jsonify({"fields": fields, "required": list(fields.keys())[:18]})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict credit score for a single customer.
    Body: JSON object with customer financial fields.
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    try:
        result = _make_prediction(data)
        log.info("Prediction: %s (conf=%.2f)", result["credit_score"], result["confidence"])
        return jsonify({"success": True, "prediction": result}), 200
    except Exception as exc:
        log.exception("Prediction failed")
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    Predict credit scores for a list of customers.
    Body: JSON array of customer objects.
    """
    data = request.get_json(force=True, silent=True)
    if not data or not isinstance(data, list):
        return jsonify({"error": "Request body must be a JSON array of customer records"}), 400
    if len(data) > 500:
        return jsonify({"error": "Batch size limited to 500 records per request"}), 400

    try:
        df = preprocess_dataframe(pd.DataFrame(data), SCALER, CAT_ENCODERS)
        pred_idxs = MODEL.predict(df)
        probas    = MODEL.predict_proba(df)

        predictions = []
        for i, (idx, proba) in enumerate(zip(pred_idxs, probas)):
            label = INV_LABEL[int(idx)]
            predictions.append({
                "index":        i,
                "credit_score": label,
                "confidence":   round(float(proba[int(idx)]), 4),
                "probabilities": {
                    "Poor":     round(float(proba[0]), 4),
                    "Standard": round(float(proba[1]), 4),
                    "Good":     round(float(proba[2]), 4),
                },
                "color": SCORE_COLORS[label],
            })

        summary = {k: sum(1 for p in predictions if p["credit_score"] == k)
                   for k in ["Good", "Standard", "Poor"]}
        log.info("Batch prediction: %d records → %s", len(data), summary)
        return jsonify({"success": True, "count": len(predictions),
                        "predictions": predictions, "summary": summary}), 200
    except Exception as exc:
        log.exception("Batch prediction failed")
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/metrics", methods=["GET"])
def metrics():
    """Return model comparison metrics for all trained models."""
    rows = []
    for model_name, vals in METADATA["metrics"].items():
        rows.append({
            "model":        model_name,
            "accuracy":     vals["Accuracy"],
            "precision":    vals["Precision"],
            "recall":       vals["Recall"],
            "f1_score":     vals["F1-Score"],
            "roc_auc":      vals["ROC-AUC"],
            "cv_accuracy":  vals["CV-Accuracy"],
            "is_best":      model_name == METADATA["best_model"],
        })
    return jsonify({"models": rows, "best_model": METADATA["best_model"]})


# ── 404 / 405 handlers ────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found", "status": 404}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed", "status": 405}), 405


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("FLASK_PORT", 5000))
    log.info("Starting Credit Scoring API on http://localhost:%d", port)
    app.run(host="0.0.0.0", port=port, debug=False)

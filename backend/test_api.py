"""
Quick smoke-test for the Flask API — imports the app, 
uses the Werkzeug test client (no real server needed).
"""
import sys
sys.path.insert(0, "backend")

from app import app

client = app.test_client()

# 1. Health
r = client.get("/health")
assert r.status_code == 200
data = r.get_json()
assert data["status"] == "ok"
print(f"[OK] /health  -> {data['model']}")

# 2. Metrics
r = client.get("/metrics")
assert r.status_code == 200
data = r.get_json()
assert len(data["models"]) == 3
print(f"[OK] /metrics -> {[m['model'] for m in data['models']]}")

# 3. Features
r = client.get("/features")
assert r.status_code == 200
print(f"[OK] /features -> {len(r.get_json()['fields'])} fields")

# 4. Single predict
payload = {
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
    "Payment_Behaviour": "Low_spent_Medium_value_payments",
}
r = client.post("/predict", json=payload)
assert r.status_code == 200
pred = r.get_json()["prediction"]
assert pred["credit_score"] in ("Good", "Standard", "Poor")
probs = pred["probabilities"]
assert abs(sum(probs.values()) - 1.0) < 0.01
print(f"[OK] /predict -> {pred['credit_score']}  (conf={pred['confidence']:.1%})")
print(f"     Probs    -> Poor={probs['Poor']:.3f}  Standard={probs['Standard']:.3f}  Good={probs['Good']:.3f}")

# 5. Batch predict (3 records)
r = client.post("/predict/batch", json=[payload, payload, payload])
assert r.status_code == 200
batch = r.get_json()
assert batch["count"] == 3
print(f"[OK] /predict/batch -> {batch['count']} predictions  summary={batch['summary']}")

# 6. 404
r = client.get("/nonexistent")
assert r.status_code == 404
print(f"[OK] 404 handler -> {r.get_json()['error']}")

print("\nAll API tests passed!")

"""
Flask backend for the Loan Eligibility Prediction web app.
Serves the HTML frontend and exposes a /predict POST endpoint.
"""

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model artifact once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'loan_model.pkl')
artifact = joblib.load(MODEL_PATH)
model = artifact['model']
FEATURE_COLS = artifact['feature_cols']


def build_reasons(data: dict, eligible: bool) -> list[str]:
    """Return human-readable feedback about key factors."""
    reasons = []

    if data['credit_history'] == 0:
        reasons.append("Poor credit history significantly reduces eligibility.")
    else:
        reasons.append("Good credit history is a strong positive factor.")

    total_income = data['applicant_income'] + data['coapplicant_income']
    income_ratio = total_income / (data['loan_amount'] + 1)
    if income_ratio < 1.5:
        reasons.append("Income may be too low relative to the requested loan amount.")
    elif income_ratio >= 3:
        reasons.append("Strong income-to-loan ratio improves your chances.")

    debt_ratio = data['existing_debts'] / (data['applicant_income'] + 1)
    if debt_ratio > 0.4:
        reasons.append("Existing debts are high compared to your income.")

    if data['years_employed'] < 2:
        reasons.append("Less than 2 years of employment history is considered a risk.")
    elif data['years_employed'] >= 5:
        reasons.append("Stable long-term employment is a positive factor.")

    if data['education'] == 1:
        reasons.append("Graduate education status is viewed favourably.")

    if not reasons:
        reasons.append("Your profile has been assessed based on all provided information.")

    return reasons[:4]  # return at most 4 reasons


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        body = request.get_json(force=True)

        # Parse and encode inputs
        gender = 1 if body.get('gender', 'male').lower() == 'male' else 0
        married = 1 if body.get('married', 'no').lower() == 'yes' else 0
        dependents = min(int(body.get('dependents', 0)), 3)
        education = 1 if body.get('education', 'graduate').lower() == 'graduate' else 0
        self_employed = 1 if body.get('self_employed', 'no').lower() == 'yes' else 0
        applicant_income = float(body.get('applicant_income', 0))
        coapplicant_income = float(body.get('coapplicant_income', 0))
        loan_amount = float(body.get('loan_amount', 1))
        loan_term = int(body.get('loan_term', 360))
        credit_history = 1 if str(body.get('credit_history', '1')) == '1' else 0
        property_area_map = {'rural': 0, 'semiurban': 1, 'urban': 2}
        property_area = property_area_map.get(body.get('property_area', 'urban').lower(), 2)
        years_employed = int(body.get('years_employed', 0))
        existing_debts = float(body.get('existing_debts', 0))

        data = {
            'gender': gender,
            'married': married,
            'dependents': dependents,
            'education': education,
            'self_employed': self_employed,
            'applicant_income': applicant_income,
            'coapplicant_income': coapplicant_income,
            'loan_amount': loan_amount,
            'loan_term': loan_term,
            'credit_history': credit_history,
            'property_area': property_area,
            'years_employed': years_employed,
            'existing_debts': existing_debts,
        }

        features = np.array([[data[col] for col in FEATURE_COLS]])
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = float(probabilities[prediction])
        eligible = bool(prediction == 1)

        reasons = build_reasons(data, eligible)

        return jsonify({
            'eligible': eligible,
            'confidence': round(confidence * 100, 1),
            'reasons': reasons,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)

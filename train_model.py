"""
Loan Eligibility Model Training Script
Generates synthetic training data and trains a RandomForestClassifier.
Run this script once to produce loan_model.pkl before starting the Flask app.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

np.random.seed(42)
N = 5000  # number of synthetic samples


def generate_dataset(n):
    """Generate a realistic synthetic loan dataset."""
    gender = np.random.choice([0, 1], n, p=[0.35, 0.65])  # 0=Female, 1=Male
    married = np.random.choice([0, 1], n, p=[0.40, 0.60])
    dependents = np.random.choice(
        [0, 1, 2, 3], n, p=[0.40, 0.25, 0.20, 0.15]
    )
    education = np.random.choice(
        [0, 1], n, p=[0.20, 0.80]
    )  # 0=Not Graduate, 1=Graduate
    self_employed = np.random.choice([0, 1], n, p=[0.86, 0.14])
    applicant_income = np.random.lognormal(
        mean=8.5, sigma=0.6, size=n
    ).astype(int)
    coapplicant_income = np.where(
        married == 1,
        np.random.lognormal(mean=7.5, sigma=0.8, size=n).astype(int),
        0,
    )
    loan_amount = np.random.lognormal(mean=5.0, sigma=0.5, size=n).astype(int)
    loan_term = np.random.choice(
        [12, 36, 60, 84, 120, 180, 240, 300, 360],
        n,
        p=[0.02, 0.03, 0.04, 0.05, 0.06, 0.10, 0.10, 0.10, 0.50],
    )
    credit_history = np.random.choice(
        [0, 1], n, p=[0.15, 0.85]
    )  # 0=Bad, 1=Good
    property_area = np.random.choice(
        [0, 1, 2], n, p=[0.30, 0.40, 0.30]
    )  # 0=Rural, 1=Semiurban, 2=Urban
    years_employed = np.random.randint(0, 35, n)
    existing_debts = np.random.lognormal(
        mean=4.0, sigma=1.0, size=n
    ).astype(int)

    # Compute eligibility based on realistic rules
    income_ratio = (applicant_income + coapplicant_income) / (loan_amount + 1)
    debt_ratio = existing_debts / (applicant_income + 1)

    score = (
        credit_history * 3.5
        + (income_ratio > 2).astype(int) * 2.0
        + (debt_ratio < 0.3).astype(int) * 1.5
        + education * 0.8
        + (years_employed > 2).astype(int) * 0.7
        + married * 0.5
        + (dependents <= 2).astype(int) * 0.3
        + (property_area != 0).astype(int) * 0.4
        - self_employed * 0.3
        + np.random.normal(0, 0.5, n)  # noise
    )

    # Set threshold at 35th percentile so ~65% are eligible
    threshold = np.percentile(score, 35)
    eligible = (score >= threshold).astype(int)

    df = pd.DataFrame({
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
        'eligible': eligible,
    })
    return df


def train():
    print("Generating synthetic training data...")
    df = generate_dataset(N)

    approval_rate = df['eligible'].mean() * 100
    print(f"Dataset: {len(df)} samples | Approval rate: {approval_rate:.1f}%")

    feature_cols = [
        'gender', 'married', 'dependents', 'education', 'self_employed',
        'applicant_income', 'coapplicant_income', 'loan_amount', 'loan_term',
        'credit_history', 'property_area', 'years_employed', 'existing_debts',
    ]

    X = df[feature_cols]
    y = df['eligible']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )

    print("Training RandomForestClassifier...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred, target_names=['Not Eligible', 'Eligible']
    ))

    # Save model and feature list together
    artifact = {
        'model': model,
        'feature_cols': feature_cols,
    }
    joblib.dump(artifact, 'loan_model.pkl')
    print("\nModel saved to loan_model.pkl")


if __name__ == '__main__':
    train()

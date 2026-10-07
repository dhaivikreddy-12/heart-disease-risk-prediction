"""Train and compare models on heart disease data with cross-validation."""
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier

df = pd.read_csv("data/heart.csv")
print(f"Loaded {len(df)} patients")
print(df.describe().round(1))

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False, eval_metric="logloss"),
}

print("\n=== Cross-validation AUC (5-fold) ===")
for name, model in models.items():
    cv = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
    model.fit(X_train_s, y_train)
    y_prob = model.predict_proba(X_test_s)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"{name:<20} CV AUC: {cv.mean():.3f} (+/- {cv.std():.3f}) | Test AUC: {auc:.3f}")

best = RandomForestClassifier(n_estimators=300, random_state=42)
best.fit(X_train_s, y_train)
importance = pd.Series(best.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n=== Feature importance (Random Forest) ===")
for feat, val in importance.items():
    print(f"{feat:<10} {val:.3f}")


def predict_patient(age, sex, cp, trestbps, chol, thalach, exang):
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(max_iter=1000).fit(scaler.transform(X), y)
    row = scaler.transform([[age, sex, cp, trestbps, chol, thalach, exang]])
    prob = model.predict_proba(row)[0][1]
    return prob


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict heart disease risk")
    for f in ["age", "sex", "cp", "trestbps", "chol", "thalach", "exang"]:
        parser.add_argument(f"--{f}", type=float, required=False)
    args = parser.parse_args()

    if args.age is not None:
        prob = predict_patient(args.age, args.sex, args.cp, args.trestbps,
                               args.chol, args.thalach, args.exang)
        print(f"\nEstimated heart disease risk: {prob:.1%}")
        print("High risk - recommend screening" if prob > 0.5 else "Lower risk")

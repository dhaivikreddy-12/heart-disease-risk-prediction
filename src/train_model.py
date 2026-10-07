"""Train and compare models on the real Heart Disease dataset."""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from xgboost import XGBClassifier

from src.load_data import load

TARGET = "target"

df = load()
print(f"Loaded {len(df)} patients, columns: {list(df.columns)}")
print(f"Disease rate: {df[TARGET].mean():.1%}")
print(f"Missing values per column: {df.isna().sum()[df.isna().sum() > 0].to_dict()}")

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "LogisticRegression": LogisticRegression(max_iter=2000),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(
        n_estimators=300, random_state=42, eval_metric="logloss",
        use_label_encoder=False, tree_method="hist",
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n=== Cross-validated AUC (5-fold) ===")
results = {}
for name, model in models.items():
    # The real Cleveland data has a handful of missing values in
    # major_vessels / thalassemia, so imputation is part of the pipeline.
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("clf", model)])
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_prob)
    results[name] = (pipe, scores.mean())
    print(f"{name:<20} CV AUC {scores.mean():.3f} (+/- {scores.std():.3f}) | Test AUC {test_auc:.3f}")

best_name = max(results, key=lambda k: results[k][1])
pipe = results[best_name][0]
y_pred = pipe.predict(X_test)
print(f"\nBest by CV: {best_name}")
print(classification_report(y_test, y_pred, target_names=["no disease", "disease"], digits=4))

clf = pipe.named_steps["clf"]
if hasattr(clf, "feature_importances_"):
    raw = clf.feature_importances_
else:
    raw = np.abs(clf.coef_.ravel())
importance = pd.Series(raw, index=X.columns).sort_values(ascending=False)
print("=== Feature importance ===")
for feat, val in importance.items():
    print(f"{feat:<26} {val:.3f}")

importance.plot(kind="barh", color="#c44e52")
plt.xlabel("Importance")
plt.title("Feature Importance - Heart Disease")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("importance.png", dpi=120)
print("Saved plot to importance.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict heart disease risk")
    for col in X.columns:
        parser.add_argument(f"--{col}", type=float, required=False)
    args = parser.parse_args()

    missing = [c for c in X.columns if getattr(args, c) is None]
    if missing:
        print(f"Missing features: {', '.join(missing)}")
        parser.print_help()
    else:
        row = pd.DataFrame([{c: getattr(args, c) for c in X.columns}])
        prob = pipe.predict_proba(row)[0][1]
        print(f"\nEstimated heart disease risk: {prob:.1%}")
        print("(Screening estimate only - not a diagnosis)")

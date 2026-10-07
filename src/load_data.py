"""Load the real UCI Heart Disease (Cleveland) dataset - 303 patients, 13 attributes.

Source: OpenML 'cleveland', the canonical UCI Cleveland heart disease records
with real clinical values (age, cholesterol, max heart rate, ...).
Cached to data/heart.csv so the repo runs offline after the first run.
"""
import os
import pandas as pd

CSV_PATH = "data/heart.csv"

RENAME = {
    "fbs": "fasting_blood_sugar",
    "restecg": "resting_ecg",
    "thalach": "max_heart_rate",
    "exang": "exercise_induced_angina",
    "oldpeak": "st_depression",
    "ca": "major_vessels",
    "thal": "thalassemia",
}


def load():
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)

    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(name="cleveland", version=1, as_frame=True, parser="auto")
    df = bunch.frame.copy()
    df = df.rename(columns=RENAME)

    # The Cleveland file uses 'num' (0 = no disease, 1-4 = severity).
    target_col = "num" if "num" in df.columns else "target"
    df["target"] = (df[target_col].astype(float) > 0).astype(int)
    df = df.drop(columns=[c for c in ["num", "dataset"] if c in df.columns])

    os.makedirs("data", exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    return df


if __name__ == "__main__":
    df = load()
    print(f"Loaded {len(df)} patients -> {CSV_PATH}")
    print(f"Disease rate: {df['target'].mean():.1%}")
    print(df.head())

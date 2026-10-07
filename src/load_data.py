"""Generate a synthetic heart-disease style dataset."""
import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(303)

n = 300
age = rng.integers(29, 77, n)
sex = rng.binomial(1, 0.65, n)
cp = rng.integers(0, 4, n)
trestbps = np.clip(rng.normal(130, 18, n), 90, 200).round(0)
chol = np.clip(rng.normal(230, 45, n), 100, 400).round(0)
thalach = np.clip(rng.normal(150, 23, n), 70, 200).round(0)
exang = rng.binomial(1, 0.33, n)

logit = (
    -2.6
    + 0.18 * (age - 50) / 10
    + 1.4 * sex
    + 2.2 * (cp >= 2)
    + 0.09 * (trestbps - 130) / 10
    + 0.07 * (chol - 230) / 20
    - 0.18 * (thalach - 150) / 10
    + 1.9 * exang
    + rng.normal(0, 0.28, size=n)
)
prob = 1 / (1 + np.exp(-logit))
target = (rng.random(n) < prob).astype(int)

df = pd.DataFrame({
    "age": age,
    "sex": sex,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "thalach": thalach,
    "exang": exang,
    "target": target,
})

os.makedirs("data", exist_ok=True)
df.to_csv("data/heart.csv", index=False)
print(f"Generated {len(df)} patients. Disease rate: {df['target'].mean():.1%}")

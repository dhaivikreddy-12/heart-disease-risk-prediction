# ❤️ Heart Disease Risk Prediction

> *"Data can't diagnose you. It can flag who deserves a closer look."*

An intermediate medical-style classification project on the **UCI Cleveland Heart Disease** dataset — 303 real patients, 13 clinical attributes. This was also the project that taught me to care about missing data: the real dataset has genuinely absent values, and my synthetic version never did.

## What this project does

- Loads the real Cleveland heart disease dataset (303 patients, 13 attributes).
- Detects and reports missing values instead of silently dropping them.
- Imputes missing values with the median inside a `Pipeline` (never before the split).
- Trains Logistic Regression, Random Forest, and XGBoost.
- Reports cross-validated AUC across 5 stratified folds, not one lucky split.
- Ranks features by importance and includes a CLI for individual screening.

## The dataset

[UCI Heart Disease (Cleveland)](https://archive.ics.uci.edu/dataset/45/heart+disease) via OpenML — 303 patients.

| Feature | Description |
|---|---|
| `age`, `sex` | Patient demographics |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol (mg/dl) |
| `max_heart_rate` | Maximum heart rate achieved |
| `st_depression` | ST depression induced by exercise |
| `major_vessels`, `thalassemia` | Vessel and thalassemia indicators |
| `target` | 1 = disease present — **target** (45.9% positive) |

The real dataset has **4 missing** `major_vessels` and **2 missing** `thalassemia` values. The code prints this out and handles it.

## How to run it

```bash
pip install -r requirements.txt

python heart.py

# Screen a single patient
python -m src.train_model --age 63 --sex 1 --cp 1 --trestbps 145 --chol 233 \
    --fasting_blood_sugar 1 --resting_ecg 2 --max_heart_rate 150 \
    --exercise_induced_angina 0 --st_depression 2.3 --slope 3 --major_vessels 0 --thalassemia 6
```

## Project structure

```
heart-disease-risk-prediction/
├── data/
│   └── heart.csv
├── src/
│   ├── load_data.py   # fetch + cache
│   └── train_model.py # impute + scale + CV + comparison
├── tests/
├── heart.py
├── requirements.txt
└── README.md
```

## What I learned

- Why cross-validation matters: one split on 303 patients can swing AUC by 0.1.
- That missing values are normal, and imputing inside the pipeline prevents leakage.
- How to read coefficients *and* feature importances, and why logistic regression is often preferred in medicine — you can explain it.
- That the features this model ranks highest match what cardiologists already prioritise, which is the best sanity check there is.

## Results

5-fold stratified cross-validation plus a 20% held-out test split:

| Model | CV AUC | Test AUC |
|---|---|---|
| **LogisticRegression** | **0.912** (±0.018) | 0.951 |
| RandomForest | 0.910 (±0.023) | **0.958** |
| XGBoost | 0.885 (±0.018) | 0.908 |

Test accuracy 0.869. Top features by importance: `thalassemia` (0.678), `cp` — chest pain type (0.543), `max_heart_rate` (0.348), `chol` (0.215).

---

*Built with Python, pandas, scikit-learn, XGBoost, matplotlib. Real clinical data, honestly measured. Screening tool only — not a diagnosis.*

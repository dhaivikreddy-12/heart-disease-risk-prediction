# ❤️ Heart Disease Risk Prediction

> *"Data can't diagnose you — but it can flag who needs a closer look."*

An intermediate medical-style classification project. Using a well-known heart disease dataset, it predicts whether a patient is at risk and — just as importantly — which clinical features matter most. I was careful to frame it as a screening tool, not a diagnosis.

## What this project does

- Loads a classic heart disease dataset (~300 patients, 13 features).
- Cleans and standardizes it.
- Trains **Logistic Regression**, **Random Forest**, and **XGBoost**.
- Compares models with accuracy, AUC, and cross-validation.
- Shows feature importance to explain the model's decisions.
- Provides a simple `predict()` function.

## The dataset

The Cleveland Heart Disease dataset, a standard benchmark in medical ML:

| Feature            | Description                     |
|--------------------|---------------------------------|
| `age`              | Age in years                    |
| `sex`              | 1 = male, 0 = female            |
| `cp`               | Chest pain type (0–3)           |
| `trestbps`         | Resting blood pressure          |
| `chol`             | Serum cholesterol (mg/dl)       |
| `thalach`          | Max heart rate achieved         |
| `exang`            | Exercise-induced angina (1/0)   |
| `target`           | 1 = disease present, 0 = absent |

## How to run it

```bash
pip install -r requirements.txt

# Full pipeline: train + compare models
python heart.py

# Single patient prediction
python heart.py --age 55 --sex 1 --cp 2 --trestbps 140 --chol 240 --thalach 150 --exang 0
```

## What I learned

- How to compare models with cross-validation instead of one lucky split.
- That logistic regression stays relevant in medicine because it's *explainable*.
- How to interpret ROC/AUC honestly.
- The responsibility that comes with medical-adjacent ML — ethics matter.

## Results

The best models reach **~0.78 AUC** on cross-validation. Chest pain type and cholesterol consistently come out as top predictors, matching what cardiologists already know — always a good sanity check.

---

*Built with Python, pandas, scikit-learn, XGBoost. Made for learning, by a student, for students.*

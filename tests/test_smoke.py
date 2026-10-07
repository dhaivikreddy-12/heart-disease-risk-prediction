import ast
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CSV = ROOT / "data" / "heart.csv"
IMPUTED_COLUMNS = ["major_vessels", "thalassemia"]


def _require_csv():
    if not CSV.exists():
        pytest.skip(f"cached dataset missing ({CSV}); run the loader once to cache it")


def _frame():
    df = pd.read_csv(CSV)
    return df.drop(columns=[c for c in ["dataset"] if c in df.columns])


def _pipeline():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )


def test_readme_and_license_exist():
    assert (ROOT / "README.md").is_file(), "README.md is missing"
    assert (ROOT / "LICENSE").is_file(), "LICENSE is missing"


def test_load_module_declares_csv_path_and_rename_map():
    from src import load_data

    assert callable(load_data.load)
    assert load_data.CSV_PATH.startswith("data/")
    assert pathlib.PurePosixPath(load_data.CSV_PATH).name == "heart.csv"
    assert load_data.RENAME["ca"] == "major_vessels"
    assert load_data.RENAME["thal"] == "thalassemia"


def test_cached_csv_shape_and_disease_rate():
    _require_csv()
    df = _frame()

    assert len(df) == 303, f"expected the Cleveland cohort, got {len(df)} rows"
    assert "target" in df.columns
    assert set(df["target"].unique()) <= {0, 1}
    rate = float(df["target"].mean())
    assert 0.40 < rate < 0.50, f"implausible disease rate {rate:.3f}"
    assert {"age", "chol", "max_heart_rate", "st_depression", "major_vessels", "thalassemia"} <= set(df.columns)


def test_dataset_really_contains_missing_clinical_values():
    """Imputation is the whole point of the project, so the NaNs must be present."""
    _require_csv()
    df = _frame()

    counts = {c: int(df[c].isna().sum()) for c in IMPUTED_COLUMNS}
    assert sum(counts.values()) > 0, f"expected real NaNs in {IMPUTED_COLUMNS}, found none"
    assert all(n > 0 for n in counts.values()), counts
    assert df[IMPUTED_COLUMNS].dtypes.apply(lambda d: d.kind == "f").all()
    assert df.drop(columns=IMPUTED_COLUMNS).isna().sum().sum() == 0, "unexpected NaNs elsewhere"


def test_imputer_pipeline_fits_and_predicts_on_full_data():
    _require_csv()
    df = _frame()
    X = df.drop(columns=["target"])
    y = df["target"]
    assert X.isna().any().any(), "nothing left for the imputer to do"

    pipe = _pipeline()
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)

    assert proba.shape == (len(df), 2)
    assert np.isfinite(proba).all()
    assert (proba >= 0).all() and (proba <= 1).all()
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert set(np.unique(pipe.predict(X))) <= {0, 1}
    assert np.isfinite(pipe.named_steps["clf"].coef_).all()


def test_classifier_alone_cannot_handle_the_nan_columns():
    """Proof the NaNs are real: without SimpleImputer, fitting must blow up."""
    _require_csv()
    from sklearn.linear_model import LogisticRegression

    df = _frame()
    X = df.drop(columns=["target"])
    y = df["target"]

    with pytest.raises(ValueError, match="NaN"):
        LogisticRegression(max_iter=2000).fit(X, y)

    imputed = X.copy()
    for col in IMPUTED_COLUMNS:
        imputed[col] = imputed[col].fillna(imputed[col].median())
    LogisticRegression(max_iter=2000).fit(imputed, y)


def test_imputer_runs_before_scaler_in_named_steps():
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    pipe = _pipeline()
    names = list(pipe.named_steps)
    assert names == ["imp", "sc", "clf"]

    kinds = [type(step) for step in pipe.named_steps.values()]
    assert kinds.index(SimpleImputer) < kinds.index(StandardScaler) < len(kinds) - 1
    assert names.index("imp") < names.index("sc")


def test_train_model_source_orders_impute_then_scale():
    """train_model.py is a script, so check its Pipeline order statically."""
    src = (ROOT / "src" / "train_model.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    steps = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Pipeline":
            entries = node.args[0].elts
            steps = [
                (e.elts[0].value if isinstance(e, ast.Tuple) else e.args[0].value)
                for e in entries
            ]
            break

    assert steps is not None, "no Pipeline(...) call found in src/train_model.py"
    assert steps[0] == "imp" and steps[1] == "sc" and steps[-1] == "clf"

    target = [
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "TARGET"
    ]
    assert target == ["target"]

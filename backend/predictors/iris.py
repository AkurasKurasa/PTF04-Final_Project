from __future__ import annotations
import pathlib
import joblib
import numpy as np

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "models"
MLP_PATH = MODELS_DIR / "iris_mlp.pkl"
FALLBACK = MODELS_DIR / "iris.pkl"
CLASSES = ["setosa", "versicolor", "virginica"]

_model = None


def _load():
    global _model
    if _model is None:
        path = MLP_PATH if MLP_PATH.exists() else FALLBACK
        if not path.exists():
            raise FileNotFoundError(f"No iris model found in {MODELS_DIR}.")
        _model = joblib.load(path)
    return _model


def predict(data: dict, files):
    model = _load()
    feats = np.array([[
        float(data["sepal_length"]),
        float(data["sepal_width"]),
        float(data["petal_length"]),
        float(data["petal_width"]),
    ]])
    pred = int(model.predict(feats)[0])
    out = {"label": CLASSES[pred]}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(feats)[0]
        out["probabilities"] = {c: float(p) for c, p in zip(CLASSES, proba)}
        out["confidence"] = float(max(proba))
    return out

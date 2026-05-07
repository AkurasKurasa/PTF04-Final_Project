from __future__ import annotations
import pathlib
import joblib
import numpy as np

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "models"
MODEL_PATH = MODELS_DIR / "boston_linear.pkl"

_model = None


def _load():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict(data: dict, files):
    model = _load()
    lstat = float(data["lstat"])
    medv = float(model.predict([[lstat]])[0])
    return {
        "value": medv * 1000.0,
        "raw_medv": medv,
        "currency": "USD",
        "slope": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "x": lstat,
    }

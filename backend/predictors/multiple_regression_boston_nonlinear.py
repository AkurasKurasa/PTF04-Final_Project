from __future__ import annotations
import pathlib
import joblib
import numpy as np

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "models"
MODEL_PATH = MODELS_DIR / "boston_poly.pkl"

_model = None


def _load():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict(data: dict, files):
    model = _load()
    rm = float(data["rm"])
    lstat = float(data["lstat"])
    ptratio = float(data["ptratio"])
    medv = float(model.predict([[rm, lstat, ptratio]])[0])
    return {
        "value": medv * 1000.0,
        "raw_medv": medv,
        "currency": "USD",
        "inputs": {"rm": rm, "lstat": lstat, "ptratio": ptratio},
    }

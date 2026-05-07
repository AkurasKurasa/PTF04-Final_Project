from __future__ import annotations
import pathlib
import joblib
import numpy as np

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "models"
MODEL_PATH = MODELS_DIR / "boston_optimized.keras"
SCALER_PATH = MODELS_DIR / "boston_optimized_scaler.pkl"

_model = None
_scaler = None


def _load():
    global _model, _scaler
    if _model is None:
        from tensorflow.keras.models import load_model
        _model = load_model(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
    return _model, _scaler


def predict(data: dict, files):
    model, scaler = _load()
    rm = float(data["rm"])
    lstat = float(data["lstat"])
    ptratio = float(data["ptratio"])
    arr = scaler.transform(np.array([[rm, lstat, ptratio]]))
    medv = float(model.predict(arr, verbose=0)[0][0])
    return {
        "value": medv * 1000.0,
        "raw_medv": medv,
        "currency": "USD",
        "inputs": {"rm": rm, "lstat": lstat, "ptratio": ptratio},
    }

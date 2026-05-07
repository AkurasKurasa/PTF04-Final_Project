from __future__ import annotations
import pathlib
import joblib
import numpy as np

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "models"
MODEL_PATH = MODELS_DIR / "boston_multi.keras"
SCALER_PATH = MODELS_DIR / "boston_multi_scaler.pkl"
FEATURES_PATH = MODELS_DIR / "boston_multi_features.pkl"

_model = None
_scaler = None
_features = None


def _load():
    global _model, _scaler, _features
    if _model is None:
        from tensorflow.keras.models import load_model
        _model = load_model(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
        _features = joblib.load(FEATURES_PATH)
    return _model, _scaler, _features


def predict(data: dict, files):
    model, scaler, features = _load()
    arr = np.array([[float(data[f]) for f in features]])
    arr_s = scaler.transform(arr)
    medv = float(model.predict(arr_s, verbose=0)[0][0])
    return {
        "value": medv * 1000.0,
        "raw_medv": medv,
        "currency": "USD",
        "inputs": {f: float(data[f]) for f in features},
    }

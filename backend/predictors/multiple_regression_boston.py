from __future__ import annotations
import pathlib
import joblib
import numpy as np

from ._onnx_helper import get_session, run

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "models"
MODEL_PATH = MODELS_DIR / "boston_multi.onnx"
SCALER_PATH = MODELS_DIR / "boston_multi_scaler.pkl"
FEATURES_PATH = MODELS_DIR / "boston_multi_features.pkl"

_scaler = None
_features = None


def _meta_load():
    global _scaler, _features
    if _scaler is None:
        _scaler = joblib.load(SCALER_PATH)
        _features = joblib.load(FEATURES_PATH)
    return _scaler, _features


def predict(data: dict, files):
    sess = get_session(MODEL_PATH)
    scaler, features = _meta_load()
    arr = np.array([[float(data[f]) for f in features]])
    arr_s = scaler.transform(arr).astype(np.float32)
    medv = float(run(sess, arr_s)[0][0])
    return {
        "value": medv * 1000.0,
        "raw_medv": medv,
        "currency": "USD",
        "inputs": {f: float(data[f]) for f in features},
    }

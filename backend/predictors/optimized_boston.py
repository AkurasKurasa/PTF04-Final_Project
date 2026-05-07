from __future__ import annotations
import pathlib
import joblib
import numpy as np

from ._onnx_helper import get_session, run

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "models"
MODEL_PATH = MODELS_DIR / "boston_optimized.onnx"
SCALER_PATH = MODELS_DIR / "boston_optimized_scaler.pkl"

_scaler = None


def _scaler_load():
    global _scaler
    if _scaler is None:
        _scaler = joblib.load(SCALER_PATH)
    return _scaler


def predict(data: dict, files):
    sess = get_session(MODEL_PATH)
    scaler = _scaler_load()
    rm = float(data["rm"])
    lstat = float(data["lstat"])
    ptratio = float(data["ptratio"])
    arr = scaler.transform(np.array([[rm, lstat, ptratio]])).astype(np.float32)
    medv = float(run(sess, arr)[0][0])
    return {
        "value": medv * 1000.0,
        "raw_medv": medv,
        "currency": "USD",
        "inputs": {"rm": rm, "lstat": lstat, "ptratio": ptratio},
    }

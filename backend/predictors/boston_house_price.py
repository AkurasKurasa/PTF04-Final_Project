from __future__ import annotations
import pathlib
import joblib
import numpy as np

from ._onnx_helper import get_session, run

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "models"
MODEL_PATH = MODELS_DIR / "boston_house_price.onnx"
SCALER_PATH = MODELS_DIR / "boston_house_price_scaler.pkl"

_scaler = None


def _scaler_load():
    global _scaler
    if _scaler is None:
        if not SCALER_PATH.exists():
            raise FileNotFoundError(f"Scaler missing: {SCALER_PATH}")
        _scaler = joblib.load(SCALER_PATH)
    return _scaler


def predict(data: dict, files):
    sess = get_session(MODEL_PATH)
    scaler = _scaler_load()

    rm = float(data["rm"])
    lstat = float(data["lstat"])
    ptratio = float(data["ptratio"])
    sample = scaler.transform(np.array([[rm, lstat, ptratio]])).astype(np.float32)
    raw = float(run(sess, sample)[0][0])
    usd = raw * 1000.0

    return {
        "value": usd,
        "raw_medv": raw,
        "currency": "USD",
        "inputs": {"rm": rm, "lstat": lstat, "ptratio": ptratio},
    }

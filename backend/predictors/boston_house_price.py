from __future__ import annotations
import pathlib

import joblib
import numpy as np

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "models"
MODEL_PATH = MODELS_DIR / "boston_house_price.keras"
SCALER_PATH = MODELS_DIR / "boston_house_price_scaler.pkl"

_model = None
_scaler = None


def _load():
    global _model, _scaler
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run the Boston notebook end-to-end first."
            )
        if not SCALER_PATH.exists():
            raise FileNotFoundError(
                f"Scaler not found at {SCALER_PATH}. Run the Boston notebook end-to-end first."
            )
        # Lazy import keras so module load doesn't block on TF
        from tensorflow.keras.models import load_model
        _model = load_model(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
    return _model, _scaler


def predict(data: dict, files):
    model, scaler = _load()

    rm = float(data["rm"])
    lstat = float(data["lstat"])
    ptratio = float(data["ptratio"])

    sample = np.array([[rm, lstat, ptratio]])
    sample_scaled = scaler.transform(sample)
    raw = float(model.predict(sample_scaled, verbose=0)[0][0])
    usd = raw * 1000.0

    return {
        "value": usd,
        "raw_medv": raw,
        "currency": "USD",
        "inputs": {"rm": rm, "lstat": lstat, "ptratio": ptratio},
    }

from __future__ import annotations
import io
import pathlib

import numpy as np
from PIL import Image

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "models"
MODEL_PATH = MODELS_DIR / "dogs_vs_cats.keras"

_model = None


def _load():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model missing at {MODEL_PATH}.")
        from tensorflow.keras.models import load_model
        _model = load_model(MODEL_PATH)
    return _model


def predict(data: dict, files):
    model = _load()
    upload = files.get("image") if files else None
    if upload is None or not getattr(upload, "filename", ""):
        raise ValueError("No image uploaded.")

    img = Image.open(io.BytesIO(upload.read())).convert("RGB").resize((32, 32))
    arr = np.asarray(img, dtype="float32") / 255.0
    p_dog = float(model.predict(np.expand_dims(arr, 0), verbose=0)[0][0])
    p_cat = 1.0 - p_dog
    return {
        "label": "dog" if p_dog > 0.5 else "cat",
        "confidence": float(max(p_dog, p_cat)),
        "scores": {"cat": p_cat, "dog": p_dog},
    }

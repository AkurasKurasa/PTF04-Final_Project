from __future__ import annotations
import io
import pathlib

import numpy as np
from PIL import Image

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "models"
MODEL_PATH = MODELS_DIR / "cifar10_cnn.keras"

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

_model = None


def _load():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run the CIFAR-10 notebook to save it."
            )
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
    arr = np.expand_dims(arr, axis=0)

    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    return {
        "label": CLASSES[idx],
        "confidence": float(probs[idx]),
        "probabilities": {c: float(p) for c, p in zip(CLASSES, probs)},
    }

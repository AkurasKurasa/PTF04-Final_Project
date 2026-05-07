from __future__ import annotations
import io
import pathlib

import numpy as np
from PIL import Image

from ._onnx_helper import get_session, run

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "models"
MODEL_PATH = MODELS_DIR / "cifar10_cnn.onnx"

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def predict(data: dict, files):
    upload = files.get("image") if files else None
    if upload is None or not getattr(upload, "filename", ""):
        raise ValueError("No image uploaded.")

    img = Image.open(io.BytesIO(upload.read())).convert("RGB").resize((32, 32))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    probs = run(get_session(MODEL_PATH), arr)[0]
    idx = int(np.argmax(probs))
    return {
        "label": CLASSES[idx],
        "confidence": float(probs[idx]),
        "probabilities": {c: float(p) for c, p in zip(CLASSES, probs)},
    }

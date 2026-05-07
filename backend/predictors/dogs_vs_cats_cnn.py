from __future__ import annotations
import io
import pathlib

import numpy as np
from PIL import Image

from ._onnx_helper import get_session, run

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "models"
MODEL_PATH = MODELS_DIR / "dogs_vs_cats.onnx"


def predict(data: dict, files):
    upload = files.get("image") if files else None
    if upload is None or not getattr(upload, "filename", ""):
        raise ValueError("No image uploaded.")

    img = Image.open(io.BytesIO(upload.read())).convert("RGB").resize((32, 32))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    p_dog = float(run(get_session(MODEL_PATH), arr)[0][0])
    p_cat = 1.0 - p_dog
    return {
        "label": "dog" if p_dog > 0.5 else "cat",
        "confidence": float(max(p_dog, p_cat)),
        "scores": {"cat": p_cat, "dog": p_dog},
    }

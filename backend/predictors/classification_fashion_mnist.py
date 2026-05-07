from __future__ import annotations
import io
import pathlib

import numpy as np
from PIL import Image, ImageOps

from ._onnx_helper import get_session, run

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "models"
MODEL_PATH = MODELS_DIR / "fashion_mnist.onnx"

CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def _preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("L")
    arr = np.asarray(img, dtype="uint8")
    h, w = arr.shape
    sample = max(2, min(h, w) // 20)
    corners = np.concatenate([
        arr[:sample, :sample].ravel(),
        arr[:sample, -sample:].ravel(),
        arr[-sample:, :sample].ravel(),
        arr[-sample:, -sample:].ravel(),
    ])
    if corners.mean() > 127:
        arr = 255 - arr

    pil = Image.fromarray(arr)
    pil = ImageOps.autocontrast(pil, cutoff=2)
    arr = np.asarray(pil, dtype="uint8")

    mask = arr > max(40, int(arr.max() * 0.3))
    if mask.any():
        ys, xs = np.where(mask)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        arr = arr[y0:y1, x0:x1]

    h, w = arr.shape
    side = max(h, w)
    pad_y = (side - h) // 2
    pad_x = (side - w) // 2
    padded = np.zeros((side, side), dtype="uint8")
    padded[pad_y:pad_y + h, pad_x:pad_x + w] = arr

    pil = Image.fromarray(padded).resize((24, 24), Image.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(pil, (2, 2))

    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(data: dict, files):
    upload = files.get("image") if files else None
    if upload is None or not getattr(upload, "filename", ""):
        raise ValueError("No image uploaded.")

    img = Image.open(io.BytesIO(upload.read()))
    arr = _preprocess(img)

    probs = run(get_session(MODEL_PATH), arr)[0]
    idx = int(np.argmax(probs))
    return {
        "label": CLASSES[idx],
        "confidence": float(probs[idx]),
        "probabilities": {c: float(p) for c, p in zip(CLASSES, probs)},
    }

from __future__ import annotations
import io
import pathlib

import numpy as np
from PIL import Image, ImageOps

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "models"
MODEL_PATH = MODELS_DIR / "fashion_mnist.keras"

CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

_model = None


def _load():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run the Fashion-MNIST notebook to save it."
            )
        from tensorflow.keras.models import load_model
        _model = load_model(MODEL_PATH)
    return _model


def _preprocess(img: Image.Image) -> np.ndarray:
    """Match the Fashion-MNIST distribution: a tightly-cropped, centred,
    light item on a black background, 28×28 grayscale."""
    img = img.convert("L")
    arr = np.asarray(img, dtype="uint8")

    # Bg detection from the 4 corners (more reliable than overall mean).
    h, w = arr.shape
    sample = max(2, min(h, w) // 20)
    corners = np.concatenate([
        arr[:sample, :sample].ravel(),
        arr[:sample, -sample:].ravel(),
        arr[-sample:, :sample].ravel(),
        arr[-sample:, -sample:].ravel(),
    ])
    bg_is_bright = corners.mean() > 127
    if bg_is_bright:
        arr = 255 - arr

    # Now: item is bright, bg is dark. Boost contrast.
    pil = Image.fromarray(arr)
    pil = ImageOps.autocontrast(pil, cutoff=2)
    arr = np.asarray(pil, dtype="uint8")

    # Threshold to find the item bounding box (anything above ~30% of max).
    mask = arr > max(40, int(arr.max() * 0.3))
    if mask.any():
        ys, xs = np.where(mask)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        arr = arr[y0:y1, x0:x1]

    # Pad to square so aspect-ratio is preserved when resized.
    h, w = arr.shape
    side = max(h, w)
    pad_y = (side - h) // 2
    pad_x = (side - w) // 2
    padded = np.zeros((side, side), dtype="uint8")
    padded[pad_y:pad_y + h, pad_x:pad_x + w] = arr

    # Resize to 28×28 with anti-aliasing, then leave a small dark border so the
    # item sits inside the 28×28 canvas (Fashion-MNIST samples have ~2px border).
    pil = Image.fromarray(padded).resize((24, 24), Image.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(pil, (2, 2))

    arr = np.asarray(canvas, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)


def predict(data: dict, files):
    model = _load()

    upload = files.get("image") if files else None
    if upload is None or not getattr(upload, "filename", ""):
        raise ValueError("No image uploaded.")

    img = Image.open(io.BytesIO(upload.read()))
    arr = _preprocess(img)

    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    return {
        "label": CLASSES[idx],
        "confidence": float(probs[idx]),
        "probabilities": {c: float(p) for c, p in zip(CLASSES, probs)},
    }

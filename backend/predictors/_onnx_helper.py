"""Tiny shared helper for loading + running ONNX models.

We deliberately avoid importing TensorFlow here so the Vercel function bundle
stays small. `onnxruntime` is the only runtime dependency."""
from __future__ import annotations
import pathlib
import threading
from typing import Any

import numpy as np

_sessions: dict[str, Any] = {}
_lock = threading.Lock()


def get_session(path: pathlib.Path):
    """Return a cached `onnxruntime.InferenceSession` for the given .onnx path."""
    key = str(path)
    with _lock:
        sess = _sessions.get(key)
        if sess is None:
            if not path.exists():
                raise FileNotFoundError(f"ONNX model missing: {path}")
            import onnxruntime as ort
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            _sessions[key] = sess
        return sess


def run(session, arr: np.ndarray) -> np.ndarray:
    """Run inference and return the first output as a numpy array."""
    name = session.get_inputs()[0].name
    outputs = session.run(None, {name: arr})
    return np.asarray(outputs[0])

from __future__ import annotations
import json
import pathlib
import re

import numpy as np

from ._onnx_helper import get_session, run

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "models"
MODEL_PATH = MODELS_DIR / "sentiment_lstm.onnx"
VOCAB_PATH = MODELS_DIR / "sentiment_vocab.json"

_word_index: dict | None = None
_maxlen = 200
_vocab_size = 20000


def _vocab_load():
    global _word_index, _maxlen, _vocab_size
    if _word_index is None:
        if not VOCAB_PATH.exists():
            raise FileNotFoundError(f"Vocab missing: {VOCAB_PATH}")
        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        _word_index = data["word_index"]
        _maxlen = int(data.get("maxlen", 200))
        _vocab_size = int(data.get("vocab_size", 20000))
    return _word_index, _maxlen, _vocab_size


def _encode(text: str) -> tuple[np.ndarray, int]:
    word_index, maxlen, vocab_size = _vocab_load()
    tokens = re.findall(r"[a-z']+", text.lower())
    ids = [1]  # <start>
    for t in tokens:
        wid = word_index.get(t)
        if wid is None or wid + 3 >= vocab_size:
            ids.append(2)  # <oov>
        else:
            ids.append(wid + 3)
    used = len(ids)
    if len(ids) >= maxlen:
        ids = ids[-maxlen:]
    else:
        ids = [0] * (maxlen - len(ids)) + ids
    return np.array([ids], dtype=np.float32), used


def predict(data: dict, files):
    text = (data.get("text") or "").strip()
    if not text:
        raise ValueError("Empty text.")

    arr, used = _encode(text)
    score = float(run(get_session(MODEL_PATH), arr)[0][0])
    label = "positive" if score > 0.5 else "negative"
    return {
        "label": label,
        "score": score,
        "polarity": (score - 0.5) * 2.0,
        "tokens_used": used,
    }

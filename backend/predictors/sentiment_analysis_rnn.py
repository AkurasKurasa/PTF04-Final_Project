from __future__ import annotations
import json
import pathlib
import re

import numpy as np

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "models"
MODEL_PATH = MODELS_DIR / "sentiment_lstm.keras"
VOCAB_PATH = MODELS_DIR / "sentiment_vocab.json"

_model = None
_word_index: dict | None = None
_maxlen = 200
_vocab_size = 20000


def _load():
    global _model, _word_index, _maxlen, _vocab_size
    if _model is None:
        if not MODEL_PATH.exists() or not VOCAB_PATH.exists():
            raise FileNotFoundError("Sentiment model or vocab missing. Run training script.")
        from tensorflow.keras.models import load_model
        _model = load_model(MODEL_PATH)
        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        _word_index = data["word_index"]
        _maxlen = int(data.get("maxlen", 200))
        _vocab_size = int(data.get("vocab_size", 20000))
    return _model, _word_index, _maxlen, _vocab_size


def _encode(text: str, word_index: dict, maxlen: int, vocab_size: int) -> tuple[np.ndarray, int]:
    """Reproduce Keras's `imdb.load_data` encoding:
       0=<pad>, 1=<start>, 2=<oov>, 3=<unused>, real words = word_index[w] + 3.
       Words above `vocab_size` map to <oov> just like during training.
    """
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
    return np.array([ids], dtype="int32"), used


def predict(data: dict, files):
    model, word_index, maxlen, vocab_size = _load()
    text = (data.get("text") or "").strip()
    if not text:
        raise ValueError("Empty text.")

    arr, used = _encode(text, word_index, maxlen, vocab_size)
    score = float(model.predict(arr, verbose=0)[0][0])
    label = "positive" if score > 0.5 else "negative"
    return {
        "label": label,
        "score": score,
        "polarity": (score - 0.5) * 2.0,
        "tokens_used": used,
    }

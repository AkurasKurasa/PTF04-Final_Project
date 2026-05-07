"""Quick-train sentiment LSTM on IMDB."""
from __future__ import annotations
import json
import pathlib

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

OUT = pathlib.Path(__file__).resolve().parent.parent / "models"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_PATH = OUT / "sentiment_lstm.keras"
VOCAB_PATH = OUT / "sentiment_vocab.json"

VOCAB_SIZE = 20000
MAXLEN = 200

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)
X_train = sequence.pad_sequences(X_train, maxlen=MAXLEN)
X_test = sequence.pad_sequences(X_test, maxlen=MAXLEN)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(MAXLEN,)),
    tf.keras.layers.Embedding(VOCAB_SIZE, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=5, batch_size=128,
          validation_data=(X_test, y_test), verbose=2)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {acc:.4f}")

model.save(MODEL_PATH)

word_index = imdb.get_word_index()
with open(VOCAB_PATH, "w", encoding="utf-8") as f:
    json.dump({"word_index": word_index, "vocab_size": VOCAB_SIZE, "maxlen": MAXLEN}, f)

print(f"Saved -> {MODEL_PATH}")
print(f"Saved -> {VOCAB_PATH}")

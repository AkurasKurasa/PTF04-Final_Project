"""Quick-train Fashion-MNIST and save to models/fashion_mnist.keras."""
from __future__ import annotations
import pathlib
import tensorflow as tf

OUT = pathlib.Path(__file__).resolve().parent.parent / "models" / "fashion_mnist.keras"
OUT.parent.mkdir(parents=True, exist_ok=True)

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=2)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {acc:.4f}")

model.save(OUT)
print(f"Saved -> {OUT} ({OUT.stat().st_size/1024:.1f} KB)")

"""Quick-train Dogs vs Cats binary CNN on CIFAR-10 cat+dog subset."""
from __future__ import annotations
import pathlib
import numpy as np
import tensorflow as tf

OUT = pathlib.Path(__file__).resolve().parent.parent / "models" / "dogs_vs_cats.keras"
OUT.parent.mkdir(parents=True, exist_ok=True)

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train, y_test = y_train.flatten(), y_test.flatten()


def keep_cats_dogs(X, y):
    mask = (y == 3) | (y == 5)
    X, y = X[mask], y[mask]
    y = (y == 5).astype("int32")  # cat=0, dog=1
    return X.astype("float32") / 255.0, y


X_train, y_train = keep_cats_dogs(X_train, y_train)
X_test, y_test = keep_cats_dogs(X_test, y_test)

augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.08),
])

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(32, 32, 3)),
    augment,
    tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=8, batch_size=128, validation_data=(X_test, y_test), verbose=2)
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {acc:.4f}")
model.save(OUT)
print(f"Saved -> {OUT} ({OUT.stat().st_size/1024:.1f} KB)")

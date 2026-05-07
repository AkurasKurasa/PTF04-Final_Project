"""Quick-train a small CIFAR-10 CNN and save to models/cifar10_cnn.keras.

Light version (5 epochs, smaller arch) so the website Demo has a working model
fast. Re-run the full CIFAR-10 notebook for a better-accuracy model later.
"""
from __future__ import annotations
import pathlib
import tensorflow as tf

OUT = pathlib.Path(__file__).resolve().parent.parent / "models" / "cifar10_cnn.keras"
OUT.parent.mkdir(parents=True, exist_ok=True)

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

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
    tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print(f"Training light CIFAR-10 model ({sum(p.numpy().size for p in model.weights):,} params)...")
model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=128,
    validation_data=(X_test, y_test),
    verbose=2,
)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {acc:.4f}")

model.save(OUT)
print(f"Saved -> {OUT} ({OUT.stat().st_size/1024/1024:.2f} MB)")

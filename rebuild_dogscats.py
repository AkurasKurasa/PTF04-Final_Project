"""Rebuild Dogs_VS_Cats_CNN.ipynb: drop anvil/colab, train binary CNN on CIFAR cat+dog subset, save to models/."""
import nbformat as nbf
import pathlib

NB_PATH = pathlib.Path("notebooks/Dogs_VS_Cats_CNN.ipynb")
nb = nbf.v4.new_notebook()
def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)

cells = [
    md("# Dogs vs Cats — Binary CNN\n\n"
       "Trains a small convolutional binary classifier on the **cat** and **dog** subset of CIFAR-10. "
       "Uses 32×32 RGB images (10 000 cats + 10 000 dogs). Saved to `models/dogs_vs_cats.keras`."),
    md("## Part 0 — Setup"),
    code(
        "import tensorflow as tf\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "from pathlib import Path\n"
        "\n"
        "print('TensorFlow', tf.__version__)"
    ),
    md("## Part 1 — Build a Cat / Dog dataset from CIFAR-10\n\n"
       "CIFAR-10 has 10 classes; we keep only `cat` (label 3) and `dog` (label 5), then relabel:\n"
       "- 0 → cat\n- 1 → dog"),
    code(
        "(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()\n"
        "y_train, y_test = y_train.flatten(), y_test.flatten()\n"
        "\n"
        "def keep_cats_dogs(X, y):\n"
        "    mask = (y == 3) | (y == 5)\n"
        "    X, y = X[mask], y[mask]\n"
        "    y = (y == 5).astype('int32')   # cat=0, dog=1\n"
        "    return X.astype('float32') / 255.0, y\n"
        "\n"
        "X_train, y_train = keep_cats_dogs(X_train, y_train)\n"
        "X_test,  y_test  = keep_cats_dogs(X_test,  y_test)\n"
        "print('Train:', X_train.shape, '  Test:', X_test.shape)"
    ),
    md("### 1.1 Inspect the data"),
    code(
        "fig, axes = plt.subplots(2, 6, figsize=(11, 4))\n"
        "labels = ['cat', 'dog']\n"
        "for i, ax in enumerate(axes.flat):\n"
        "    ax.imshow(X_train[i])\n"
        "    ax.set_title(labels[int(y_train[i])])\n"
        "    ax.axis('off')\n"
        "plt.tight_layout(); plt.show()"
    ),
    md("## Part 2 — Build the CNN"),
    code(
        "augment = tf.keras.Sequential([\n"
        "    tf.keras.layers.RandomFlip('horizontal'),\n"
        "    tf.keras.layers.RandomRotation(0.08),\n"
        "])\n"
        "\n"
        "model = tf.keras.Sequential([\n"
        "    tf.keras.layers.Input(shape=(32, 32, 3)),\n"
        "    augment,\n"
        "    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),\n"
        "    tf.keras.layers.BatchNormalization(),\n"
        "    tf.keras.layers.MaxPooling2D(),\n"
        "    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),\n"
        "    tf.keras.layers.BatchNormalization(),\n"
        "    tf.keras.layers.MaxPooling2D(),\n"
        "    tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),\n"
        "    tf.keras.layers.BatchNormalization(),\n"
        "    tf.keras.layers.GlobalAveragePooling2D(),\n"
        "    tf.keras.layers.Dropout(0.3),\n"
        "    tf.keras.layers.Dense(1, activation='sigmoid'),  # binary\n"
        "])\n"
        "model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])\n"
        "model.summary()"
    ),
    md("## Part 3 — Train"),
    code(
        "history = model.fit(\n"
        "    X_train, y_train,\n"
        "    epochs=8,\n"
        "    batch_size=128,\n"
        "    validation_data=(X_test, y_test),\n"
        "    verbose=1,\n"
        ")"
    ),
    md("### 3.1 Plot history"),
    code(
        "fig, ax = plt.subplots(1, 2, figsize=(12, 4))\n"
        "ax[0].plot(history.history['loss'], label='train')\n"
        "ax[0].plot(history.history['val_loss'], label='val')\n"
        "ax[0].set_title('Loss'); ax[0].legend()\n"
        "ax[1].plot(history.history['accuracy'], label='train')\n"
        "ax[1].plot(history.history['val_accuracy'], label='val')\n"
        "ax[1].set_title('Accuracy'); ax[1].legend()\n"
        "plt.tight_layout(); plt.show()"
    ),
    md("## Part 4 — Evaluate"),
    code(
        "loss, acc = model.evaluate(X_test, y_test, verbose=0)\n"
        "print(f'Test accuracy: {acc:.4f}')"
    ),
    md("## Part 5 — Save the Model"),
    code(
        "MODELS_DIR = Path('models')\n"
        "MODELS_DIR.mkdir(parents=True, exist_ok=True)\n"
        "MODEL_PATH = MODELS_DIR / 'dogs_vs_cats.keras'\n"
        "model.save(MODEL_PATH)\n"
        "print('Saved ->', MODEL_PATH, f'({MODEL_PATH.stat().st_size/1024:.1f} KB)')"
    ),
    md("## Part 6 — Predict Helper"),
    code(
        "from PIL import Image\n"
        "\n"
        "def predict_pet(img_or_path):\n"
        "    if isinstance(img_or_path, (str, Path)):\n"
        "        img = Image.open(img_or_path).convert('RGB')\n"
        "    else:\n"
        "        img = Image.fromarray(np.asarray(img_or_path)).convert('RGB')\n"
        "    img = img.resize((32, 32))\n"
        "    arr = np.asarray(img, dtype='float32') / 255.0\n"
        "    p_dog = float(model.predict(np.expand_dims(arr, 0), verbose=0)[0][0])\n"
        "    return ('dog' if p_dog > 0.5 else 'cat'), p_dog\n"
        "\n"
        "label, p = predict_pet(X_test[0])\n"
        "print(f'Sample 0 -> {label} (p_dog={p:.2%})')"
    ),
]
nb.cells = cells
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.13"}}
nbf.write(nb, str(NB_PATH))
print(f"Wrote {NB_PATH} ({len(cells)} cells)")

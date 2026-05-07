"""Rebuild CNN_Cifar10.ipynb: drop colab, fix class names, save model to models/, add prediction helper."""
import nbformat as nbf
import pathlib

NB_PATH = pathlib.Path("notebooks/CNN_Cifar10.ipynb")
nb = nbf.v4.new_notebook()

def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)

cells = [
    md("# CIFAR-10 Image Classifier\n\n"
       "Convolutional neural network trained on CIFAR-10 (10 image classes, 32×32 RGB). "
       "Uses data augmentation, batch normalisation, and a learning-rate scheduler. "
       "Final model is saved to `models/cifar10_cnn.keras` for the website Demo."),

    md("## Part 0 — Setup"),
    code(
        "import tensorflow as tf\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "from pathlib import Path\n"
        "\n"
        "from tensorflow.keras.datasets import cifar10\n"
        "\n"
        "print('TensorFlow', tf.__version__)"
    ),

    md("### Class labels (CIFAR-10 has 10 categories)"),
    code(
        "class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',\n"
        "               'dog', 'frog', 'horse', 'ship', 'truck']"
    ),

    md("## Part 1 — Load & Prepare Data"),
    code(
        "(X_train, y_train), (X_test, y_test) = cifar10.load_data()\n"
        "print('Train:', X_train.shape, '  Test:', X_test.shape)"
    ),

    md("### Normalise pixel values to [0, 1]"),
    code(
        "X_train = X_train.astype('float32') / 255.0\n"
        "X_test  = X_test.astype('float32') / 255.0"
    ),

    md("### Inspect a sample"),
    code(
        "fig, axes = plt.subplots(2, 5, figsize=(10, 4))\n"
        "for i, ax in enumerate(axes.flat):\n"
        "    ax.imshow(X_train[i])\n"
        "    ax.set_title(class_names[int(y_train[i])])\n"
        "    ax.axis('off')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),

    md("## Part 2 — Build the Model\n\n"
       "Three convolutional blocks → global average pooling → dense classifier. "
       "Augmentation layer is applied during training only."),
    code(
        "data_augmentation = tf.keras.Sequential([\n"
        "    tf.keras.layers.RandomFlip('horizontal'),\n"
        "    tf.keras.layers.RandomRotation(0.1),\n"
        "    tf.keras.layers.RandomZoom(0.1),\n"
        "])\n"
        "\n"
        "model = tf.keras.Sequential([\n"
        "    tf.keras.layers.Input(shape=(32, 32, 3)),\n"
        "    data_augmentation,\n"
        "\n"
        "    # Block 1\n"
        "    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),\n"
        "    tf.keras.layers.BatchNormalization(),\n"
        "    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),\n"
        "    tf.keras.layers.BatchNormalization(),\n"
        "    tf.keras.layers.MaxPooling2D(),\n"
        "\n"
        "    # Block 2\n"
        "    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),\n"
        "    tf.keras.layers.BatchNormalization(),\n"
        "    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),\n"
        "    tf.keras.layers.BatchNormalization(),\n"
        "    tf.keras.layers.MaxPooling2D(),\n"
        "\n"
        "    # Block 3\n"
        "    tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),\n"
        "    tf.keras.layers.BatchNormalization(),\n"
        "    tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),\n"
        "    tf.keras.layers.BatchNormalization(),\n"
        "    tf.keras.layers.MaxPooling2D(),\n"
        "\n"
        "    # Head\n"
        "    tf.keras.layers.GlobalAveragePooling2D(),\n"
        "    tf.keras.layers.Dense(256, activation='relu'),\n"
        "    tf.keras.layers.BatchNormalization(),\n"
        "    tf.keras.layers.Dropout(0.4),\n"
        "    tf.keras.layers.Dense(10, activation='softmax'),\n"
        "])\n"
        "\n"
        "model.compile(\n"
        "    optimizer='adam',\n"
        "    loss='sparse_categorical_crossentropy',\n"
        "    metrics=['accuracy'],\n"
        ")\n"
        "model.summary()"
    ),

    md("## Part 3 — Train\n\n"
       "ReduceLROnPlateau drops the learning rate when validation loss stops improving. "
       "EarlyStopping cuts the run short if no progress is made. "
       "Tip: lower `EPOCHS` for a quick demo, raise it for accuracy."),
    code(
        "EPOCHS = 15  # bump to 30+ for higher accuracy\n"
        "\n"
        "reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(\n"
        "    monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6\n"
        ")\n"
        "early_stop = tf.keras.callbacks.EarlyStopping(\n"
        "    monitor='val_loss', patience=6, restore_best_weights=True\n"
        ")\n"
        "\n"
        "history = model.fit(\n"
        "    X_train, y_train,\n"
        "    epochs=EPOCHS,\n"
        "    validation_data=(X_test, y_test),\n"
        "    callbacks=[reduce_lr, early_stop],\n"
        "    verbose=1,\n"
        ")"
    ),

    md("### 3.1 Plot training history"),
    code(
        "fig, ax = plt.subplots(1, 2, figsize=(12, 4))\n"
        "ax[0].plot(history.history['loss'], label='train')\n"
        "ax[0].plot(history.history['val_loss'], label='val')\n"
        "ax[0].set_title('Loss'); ax[0].set_xlabel('Epoch'); ax[0].legend()\n"
        "ax[1].plot(history.history['accuracy'], label='train')\n"
        "ax[1].plot(history.history['val_accuracy'], label='val')\n"
        "ax[1].set_title('Accuracy'); ax[1].set_xlabel('Epoch'); ax[1].legend()\n"
        "plt.tight_layout(); plt.show()"
    ),

    md("## Part 4 — Evaluation"),
    code(
        "test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)\n"
        "print(f'Test loss     : {test_loss:.4f}')\n"
        "print(f'Test accuracy : {test_accuracy:.4f}')"
    ),

    md("### 4.1 Inspect predictions on test samples"),
    code(
        "preds = model.predict(X_test[:10], verbose=0)\n"
        "fig, axes = plt.subplots(2, 5, figsize=(12, 5))\n"
        "for i, ax in enumerate(axes.flat):\n"
        "    ax.imshow(X_test[i])\n"
        "    pred_class = class_names[int(np.argmax(preds[i]))]\n"
        "    true_class = class_names[int(y_test[i])]\n"
        "    ok = pred_class == true_class\n"
        "    ax.set_title(f'{pred_class}\\n(true: {true_class})',\n"
        "                 color='green' if ok else 'red', fontsize=9)\n"
        "    ax.axis('off')\n"
        "plt.tight_layout(); plt.show()"
    ),

    md("## Part 5 — Persist the Model\n\n"
       "Saved to `models/cifar10_cnn.keras` so the website backend can load it for the Demo."),
    code(
        "MODELS_DIR = Path('models')\n"
        "MODELS_DIR.mkdir(parents=True, exist_ok=True)\n"
        "\n"
        "MODEL_PATH = MODELS_DIR / 'cifar10_cnn.keras'\n"
        "model.save(MODEL_PATH)\n"
        "\n"
        "print('Saved ->', MODEL_PATH, f'({MODEL_PATH.stat().st_size/1024:.1f} KB)')"
    ),

    md("## Part 6 — Prediction Helper\n\n"
       "Reusable function that loads any image, resizes to 32×32, and returns a class label "
       "with confidence. The website backend imports this logic via `backend/predictors/cnn_cifar10.py`."),
    code(
        "from PIL import Image\n"
        "\n"
        "def predict_image(img_or_path):\n"
        "    \"\"\"Accepts a file path, PIL.Image, or numpy array. Returns (label, confidence, all_probs).\"\"\"\n"
        "    if isinstance(img_or_path, (str, Path)):\n"
        "        img = Image.open(img_or_path).convert('RGB')\n"
        "    elif isinstance(img_or_path, Image.Image):\n"
        "        img = img_or_path.convert('RGB')\n"
        "    else:\n"
        "        img = Image.fromarray(np.asarray(img_or_path)).convert('RGB')\n"
        "\n"
        "    img = img.resize((32, 32))\n"
        "    arr = np.asarray(img, dtype='float32') / 255.0\n"
        "    arr = np.expand_dims(arr, axis=0)\n"
        "    probs = model.predict(arr, verbose=0)[0]\n"
        "    idx = int(np.argmax(probs))\n"
        "    return class_names[idx], float(probs[idx]), {c: float(p) for c, p in zip(class_names, probs)}\n"
        "\n"
        "label, conf, _ = predict_image(X_test[0])\n"
        "print(f'Test sample 0 -> {label} ({conf:.2%})')"
    ),
]

nb.cells = cells
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
}

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(NB_PATH))
print(f"Wrote {NB_PATH} ({len(cells)} cells)")

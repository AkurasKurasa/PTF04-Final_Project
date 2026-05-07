"""Rebuild Classification_Fashion_MNIST.ipynb: drop license boilerplate, fix paths, save model to models/."""
import nbformat as nbf
import pathlib

NB_PATH = pathlib.Path("notebooks/Classification_Fashion_MNIST.ipynb")
nb = nbf.v4.new_notebook()

def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)

cells = [
    md("# Fashion-MNIST Image Classification\n\n"
       "Trains a small fully-connected neural network on Fashion-MNIST: 70 000 grayscale 28×28 images "
       "spread across 10 clothing categories. Final model is saved to `models/fashion_mnist.keras` for "
       "the website Demo."),

    md("## Part 0 — Setup"),
    code(
        "import tensorflow as tf\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "from pathlib import Path\n"
        "\n"
        "print('TensorFlow', tf.__version__)"
    ),

    md("## Part 1 — Load the Fashion-MNIST Dataset\n\n"
       "Bundled with Keras: 60 000 train + 10 000 test grayscale images, 28×28."),
    code(
        "fashion_mnist = tf.keras.datasets.fashion_mnist\n"
        "(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()\n"
        "print('Train:', train_images.shape, '  Test:', test_images.shape)"
    ),

    code(
        "class_names = [\n"
        "    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',\n"
        "    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot',\n"
        "]"
    ),

    md("### 1.1 Inspect the data"),
    code(
        "fig, axes = plt.subplots(2, 5, figsize=(10, 4))\n"
        "for i, ax in enumerate(axes.flat):\n"
        "    ax.imshow(train_images[i], cmap='gray')\n"
        "    ax.set_title(class_names[int(train_labels[i])], fontsize=9)\n"
        "    ax.axis('off')\n"
        "plt.tight_layout(); plt.show()"
    ),

    md("## Part 2 — Preprocess\n\n"
       "Pixels are 0–255 ints; scale to floats in [0, 1] for stable training."),
    code(
        "train_images = train_images.astype('float32') / 255.0\n"
        "test_images  = test_images.astype('float32') / 255.0"
    ),

    md("## Part 3 — Build the Model\n\n"
       "Tiny, fast network: flatten → Dense(128) → Dense(10). Outputs softmax probabilities directly."),
    code(
        "model = tf.keras.Sequential([\n"
        "    tf.keras.layers.Input(shape=(28, 28)),\n"
        "    tf.keras.layers.Flatten(),\n"
        "    tf.keras.layers.Dense(128, activation='relu'),\n"
        "    tf.keras.layers.Dropout(0.2),\n"
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

    md("## Part 4 — Train"),
    code(
        "EPOCHS = 10\n"
        "\n"
        "history = model.fit(\n"
        "    train_images, train_labels,\n"
        "    epochs=EPOCHS,\n"
        "    validation_data=(test_images, test_labels),\n"
        "    verbose=1,\n"
        ")"
    ),

    md("### 4.1 Plot training history"),
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

    md("## Part 5 — Evaluation"),
    code(
        "test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)\n"
        "print(f'Test loss     : {test_loss:.4f}')\n"
        "print(f'Test accuracy : {test_acc:.4f}')"
    ),

    md("### 5.1 Visualise predictions on test samples"),
    code(
        "preds = model.predict(test_images[:15], verbose=0)\n"
        "fig, axes = plt.subplots(3, 5, figsize=(12, 7))\n"
        "for i, ax in enumerate(axes.flat):\n"
        "    ax.imshow(test_images[i], cmap='gray')\n"
        "    pred_idx = int(np.argmax(preds[i]))\n"
        "    true_idx = int(test_labels[i])\n"
        "    ok = pred_idx == true_idx\n"
        "    title = f'{class_names[pred_idx]}\\n(true: {class_names[true_idx]})'\n"
        "    ax.set_title(title, color='green' if ok else 'red', fontsize=8)\n"
        "    ax.axis('off')\n"
        "plt.tight_layout(); plt.show()"
    ),

    md("## Part 6 — Persist the Model\n\n"
       "Saves to `models/fashion_mnist.keras` so the website backend can load it on demand."),
    code(
        "MODELS_DIR = Path('models')\n"
        "MODELS_DIR.mkdir(parents=True, exist_ok=True)\n"
        "\n"
        "MODEL_PATH = MODELS_DIR / 'fashion_mnist.keras'\n"
        "model.save(MODEL_PATH)\n"
        "print('Saved ->', MODEL_PATH, f'({MODEL_PATH.stat().st_size/1024:.1f} KB)')"
    ),

    md("## Part 7 — Prediction Helper\n\n"
       "Reusable function. Accepts file path / PIL image / numpy array, returns label + confidence."),
    code(
        "from PIL import Image, ImageOps\n"
        "\n"
        "def predict_image(img_or_path):\n"
        "    \"\"\"Classify an image into one of the 10 Fashion-MNIST classes.\n"
        "\n"
        "    Converts to grayscale, resizes to 28x28, inverts if needed (Fashion-MNIST is\n"
        "    light items on dark bg).\n"
        "    \"\"\"\n"
        "    if isinstance(img_or_path, (str, Path)):\n"
        "        img = Image.open(img_or_path)\n"
        "    elif isinstance(img_or_path, Image.Image):\n"
        "        img = img_or_path\n"
        "    else:\n"
        "        img = Image.fromarray(np.asarray(img_or_path))\n"
        "\n"
        "    img = img.convert('L').resize((28, 28))\n"
        "    arr = np.asarray(img, dtype='float32')\n"
        "    # If background looks bright, invert (Fashion-MNIST style is light item, dark bg)\n"
        "    if arr.mean() > 127:\n"
        "        arr = 255 - arr\n"
        "    arr = arr / 255.0\n"
        "    arr = np.expand_dims(arr, axis=0)\n"
        "    probs = model.predict(arr, verbose=0)[0]\n"
        "    idx = int(np.argmax(probs))\n"
        "    return class_names[idx], float(probs[idx]), {c: float(p) for c, p in zip(class_names, probs)}\n"
        "\n"
        "label, conf, _ = predict_image(test_images[0])\n"
        "print(f'Test sample 0 -> {label} ({conf:.2%})')"
    ),
]

nb.cells = cells
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
}
nbf.write(nb, str(NB_PATH))
print(f"Wrote {NB_PATH} ({len(cells)} cells)")

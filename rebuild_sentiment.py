"""Rebuild Sentiment_Analysis_RNN.ipynb: drop anvil, train LSTM on IMDB, save model + tokenizer mapping."""
import nbformat as nbf
import pathlib

NB_PATH = pathlib.Path("notebooks/Sentiment_Analysis_RNN.ipynb")
nb = nbf.v4.new_notebook()
def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)

cells = [
    md("# Sentiment Analysis — LSTM\n\n"
       "Recurrent neural network trained on the **IMDB movie reviews** corpus to classify text as "
       "**positive** or **negative**. Model + word index saved to `models/` for the website Demo."),
    md("## Part 0 — Setup"),
    code(
        "import tensorflow as tf\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import json, pickle\n"
        "from pathlib import Path\n"
        "\n"
        "from tensorflow.keras.datasets import imdb\n"
        "from tensorflow.keras.preprocessing import sequence\n"
        "\n"
        "print('TensorFlow', tf.__version__)"
    ),
    md("## Part 1 — Load IMDB Reviews\n\n"
       "Keeps the top 20 000 most frequent words. Reviews are already tokenised as integer sequences."),
    code(
        "VOCAB_SIZE = 20000\n"
        "MAXLEN = 200\n"
        "\n"
        "(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)\n"
        "print('Train:', X_train.shape, '  Test:', X_test.shape)"
    ),
    md("### 1.1 Pad sequences to fixed length"),
    code(
        "X_train = sequence.pad_sequences(X_train, maxlen=MAXLEN)\n"
        "X_test  = sequence.pad_sequences(X_test,  maxlen=MAXLEN)\n"
        "X_train.shape"
    ),
    md("### 1.2 Decode a sample (sanity check)"),
    code(
        "word_index = imdb.get_word_index()\n"
        "index_to_word = {v + 3: k for k, v in word_index.items()}\n"
        "index_to_word[0] = '<pad>'\n"
        "index_to_word[1] = '<start>'\n"
        "index_to_word[2] = '<oov>'\n"
        "decoded = ' '.join(index_to_word.get(i, '?') for i in X_train[0] if i != 0)\n"
        "print(decoded[:300], '...')\n"
        "print('Sentiment:', 'positive' if y_train[0] == 1 else 'negative')"
    ),
    md("## Part 2 — Build the LSTM"),
    code(
        "model = tf.keras.Sequential([\n"
        "    tf.keras.layers.Input(shape=(MAXLEN,)),\n"
        "    tf.keras.layers.Embedding(VOCAB_SIZE, 64),\n"
        "    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),\n"
        "    tf.keras.layers.Dropout(0.3),\n"
        "    tf.keras.layers.Dense(32, activation='relu'),\n"
        "    tf.keras.layers.Dense(1, activation='sigmoid'),\n"
        "])\n"
        "model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])\n"
        "model.summary()"
    ),
    md("## Part 3 — Train"),
    code(
        "history = model.fit(\n"
        "    X_train, y_train,\n"
        "    epochs=3,\n"
        "    batch_size=128,\n"
        "    validation_data=(X_test, y_test),\n"
        "    verbose=1,\n"
        ")"
    ),
    md("### 3.1 Plot training history"),
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
    md("## Part 5 — Persist Model + Vocabulary"),
    code(
        "MODELS_DIR = Path('models')\n"
        "MODELS_DIR.mkdir(parents=True, exist_ok=True)\n"
        "MODEL_PATH = MODELS_DIR / 'sentiment_lstm.keras'\n"
        "VOCAB_PATH = MODELS_DIR / 'sentiment_vocab.json'\n"
        "\n"
        "model.save(MODEL_PATH)\n"
        "with open(VOCAB_PATH, 'w', encoding='utf-8') as f:\n"
        "    json.dump({\n"
        "        'word_index': word_index,\n"
        "        'vocab_size': VOCAB_SIZE,\n"
        "        'maxlen': MAXLEN,\n"
        "    }, f)\n"
        "print('Saved ->', MODEL_PATH, VOCAB_PATH)"
    ),
    md("## Part 6 — Predict Helper"),
    code(
        "import re\n"
        "\n"
        "def encode(text: str):\n"
        "    tokens = re.findall(r'[a-z\\']+', text.lower())\n"
        "    ids = [word_index.get(t, 2) + 3 for t in tokens]\n"
        "    return sequence.pad_sequences([ids], maxlen=MAXLEN)\n"
        "\n"
        "def predict_sentiment(text: str):\n"
        "    p = float(model.predict(encode(text), verbose=0)[0][0])\n"
        "    return ('positive' if p > 0.5 else 'negative'), p\n"
        "\n"
        "for sample in [\n"
        "    'Absolutely loved this movie, the performances were stunning!',\n"
        "    'Worst film I have seen in years. Boring and cliché.',\n"
        "    'It was fine. Some scenes worked, some did not.',\n"
        "]:\n"
        "    label, p = predict_sentiment(sample)\n"
        "    print(f'{p:.3f} -> {label.upper():8s} | {sample}')"
    ),
]
nb.cells = cells
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.13"}}
nbf.write(nb, str(NB_PATH))
print(f"Wrote {NB_PATH} ({len(cells)} cells)")

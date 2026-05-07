"""Rebuild Boston_House_Price.ipynb: drop anvil + pip, fix bugs, add missing logic + model save."""
import nbformat as nbf
import pathlib

NB_PATH = pathlib.Path("notebooks/Boston_House_Price.ipynb")

nb = nbf.v4.new_notebook()

def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)

cells = [
    md("# Boston Housing Price Prediction\n\n"
       "Neural network regression on the classic Boston Housing dataset using TensorFlow / Keras. "
       "Predicts median house price from `RM`, `LSTAT`, and `PTRATIO`. "
       "Trained model + scaler are persisted to `models/` so the website can serve a live GUI."),

    md("## Part 0 — Setup"),
    md("### 0.1 Imports"),
    code(
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "import joblib\n"
        "from pathlib import Path\n"
        "\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n"
        "\n"
        "from tensorflow.keras.models import Sequential\n"
        "from tensorflow.keras.layers import Dense, Input\n"
        "from tensorflow.keras.callbacks import EarlyStopping"
    ),

    md("### 0.2 Early stopping callback"),
    code(
        "early_stop = EarlyStopping(\n"
        "    monitor='val_loss',\n"
        "    patience=10,\n"
        "    restore_best_weights=True\n"
        ")"
    ),

    md("## Part 1 — Load and Inspect the Dataset"),
    md("### 1.1 Load Boston housing CSV"),
    code(
        "df = pd.read_csv('datasets/Boston1.csv')\n"
        "df.rename(columns={'medv': 'price'}, inplace=True)\n"
        "df.head()"
    ),

    md("### 1.2 Dataset info & null check"),
    code("df.info()"),
    code("df.isnull().sum()"),

    md("**Q1.** Rows × Columns → `506 × 14`  \n**Q2.** Columns with nulls → `0`"),

    md("## Part 2 — Feature Selection (Correlation)"),
    md("### 2.1 Correlation matrix"),
    code("corr = df.corr()\ncorr"),

    md("### 2.2 Heatmap"),
    code(
        "plt.figure(figsize=(12, 10))\n"
        "sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f')\n"
        "plt.title('Feature Correlation Heatmap')\n"
        "plt.show()"
    ),

    md("### 2.3 Sort correlation with price"),
    code("corr['price'].sort_values(ascending=False)"),

    md("**Top features by |correlation|:** `lstat`, `ptratio`, `rm`."),

    md("## Part 3 — Prepare the Data"),
    md("### 3.1 Features and target"),
    code(
        "X = df[['rm', 'lstat', 'ptratio']]\n"
        "y = df['price']"
    ),

    md("### 3.2 Train / test split"),
    code(
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    X, y,\n"
        "    test_size=0.2,\n"
        "    random_state=42\n"
        ")\n"
        "print('Train:', X_train.shape, '  Test:', X_test.shape)"
    ),

    md("### 3.3 Feature scaling"),
    code(
        "scaler = StandardScaler()\n"
        "X_train_s = scaler.fit_transform(X_train)\n"
        "X_test_s  = scaler.transform(X_test)"
    ),

    md("## Part 4 — Build a Neural Network"),
    md("### 4.1 Define model architecture"),
    code(
        "model = Sequential([\n"
        "    Input(shape=(X_train_s.shape[1],)),\n"
        "    Dense(32, activation='relu'),\n"
        "    Dense(16, activation='relu'),\n"
        "    Dense(1)\n"
        "])\n"
        "model.summary()"
    ),

    md("### 4.2 Compile"),
    code(
        "model.compile(\n"
        "    optimizer='adam',\n"
        "    loss='mse',\n"
        "    metrics=['mae']\n"
        ")"
    ),

    md("## Part 5 — Train the Model"),
    md("### 5.1 Train 100 epochs"),
    code(
        "history_100 = model.fit(\n"
        "    X_train_s, y_train,\n"
        "    validation_split=0.2,\n"
        "    epochs=100,\n"
        "    batch_size=16,\n"
        "    verbose=0,\n"
        "    callbacks=[early_stop]\n"
        ")\n"
        "print('Final train loss:', history_100.history['loss'][-1])\n"
        "print('Final val loss:  ', history_100.history['val_loss'][-1])"
    ),

    md("### 5.2 Continue training for 300 epochs (fresh model)"),
    code(
        "model_300 = Sequential([\n"
        "    Input(shape=(X_train_s.shape[1],)),\n"
        "    Dense(32, activation='relu'),\n"
        "    Dense(16, activation='relu'),\n"
        "    Dense(1)\n"
        "])\n"
        "model_300.compile(optimizer='adam', loss='mse', metrics=['mae'])\n"
        "\n"
        "history_300 = model_300.fit(\n"
        "    X_train_s, y_train,\n"
        "    validation_split=0.2,\n"
        "    epochs=300,\n"
        "    batch_size=16,\n"
        "    verbose=0,\n"
        "    callbacks=[early_stop]\n"
        ")\n"
        "print('Final train loss:', history_300.history['loss'][-1])\n"
        "print('Final val loss:  ', history_300.history['val_loss'][-1])"
    ),

    md("### 5.3 Plot training vs validation loss"),
    code(
        "fig, ax = plt.subplots(1, 2, figsize=(12, 4))\n"
        "ax[0].plot(history_100.history['loss'], label='train')\n"
        "ax[0].plot(history_100.history['val_loss'], label='val')\n"
        "ax[0].set_title('100 epochs'); ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('Loss'); ax[0].legend()\n"
        "ax[1].plot(history_300.history['loss'], label='train')\n"
        "ax[1].plot(history_300.history['val_loss'], label='val')\n"
        "ax[1].set_title('300 epochs'); ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('Loss'); ax[1].legend()\n"
        "plt.tight_layout(); plt.show()"
    ),

    md("## Part 6 — Model Evaluation"),
    md("### 6.1 Predict on test set"),
    code(
        "y_pred_100 = model.predict(X_test_s, verbose=0)\n"
        "y_pred_300 = model_300.predict(X_test_s, verbose=0)"
    ),

    md("### 6.2 Compute metrics"),
    code(
        "def metrics(y_true, y_pred):\n"
        "    mae  = mean_absolute_error(y_true, y_pred)\n"
        "    mse  = mean_squared_error(y_true, y_pred)\n"
        "    rmse = np.sqrt(mse)\n"
        "    r2   = r2_score(y_true, y_pred)\n"
        "    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}\n"
        "\n"
        "results = pd.DataFrame({\n"
        "    '100 epochs': metrics(y_test, y_pred_100),\n"
        "    '300 epochs': metrics(y_test, y_pred_300),\n"
        "}).T\n"
        "results"
    ),

    md("## Part 7 — Single Prediction"),
    md("### 7.1 Sample house features (RM, LSTAT, PTRATIO)"),
    code(
        "sample = pd.DataFrame([[6.0, 12.0, 15.0]], columns=['rm', 'lstat', 'ptratio'])\n"
        "sample"
    ),

    md("### 7.2 Predict price"),
    code(
        "sample_scaled  = scaler.transform(sample)\n"
        "predicted      = model_300.predict(sample_scaled, verbose=0)[0][0]\n"
        "predicted_usd  = predicted * 1000  # MEDV is in $1000s\n"
        "print(f'Predicted (raw MEDV): {predicted:.2f}')\n"
        "print(f'Predicted price (USD): ${predicted_usd:,.2f}')"
    ),

    md("### 7.3 Multiple sample predictions"),
    code(
        "samples = pd.DataFrame(\n"
        "    [\n"
        "        [6.0, 12.0, 15.0],\n"
        "        [12.7, 6.5, 53.5],\n"
        "        [4.2, 67.5, 19.5],\n"
        "    ],\n"
        "    columns=['rm', 'lstat', 'ptratio'],\n"
        ")\n"
        "samples_scaled = scaler.transform(samples)\n"
        "preds          = model_300.predict(samples_scaled, verbose=0).flatten()\n"
        "samples['predicted_price_usd'] = (preds * 1000).round(2)\n"
        "samples"
    ),

    md("## Part 8 — Persist Model for Web GUI\n\n"
       "Save the trained model + scaler so the website backend can load them and serve "
       "live predictions through the in-page form (no Anvil)."),
    code(
        "MODELS_DIR = Path('models')\n"
        "MODELS_DIR.mkdir(parents=True, exist_ok=True)\n"
        "\n"
        "model_300.save(MODELS_DIR / 'boston_house_price.keras')\n"
        "joblib.dump(scaler, MODELS_DIR / 'boston_house_price_scaler.pkl')\n"
        "\n"
        "print('Saved:')\n"
        "print(' -', MODELS_DIR / 'boston_house_price.keras')\n"
        "print(' -', MODELS_DIR / 'boston_house_price_scaler.pkl')"
    ),

    md("## Part 9 — Prediction Helper\n\n"
       "Reusable function the website backend calls."),
    code(
        "def predict_price(rm: float, lstat: float, ptratio: float) -> float:\n"
        "    \"\"\"Predict house price in USD given RM, LSTAT, PTRATIO.\"\"\"\n"
        "    sample        = np.array([[rm, lstat, ptratio]])\n"
        "    sample_scaled = scaler.transform(sample)\n"
        "    pred          = model_300.predict(sample_scaled, verbose=0)[0][0]\n"
        "    return float(pred * 1000)\n"
        "\n"
        "predict_price(6.0, 12.0, 15.0)"
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

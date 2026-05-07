"""Rebuild the Linear / Multiple / Nonlinear / Optimized Boston notebooks.

Each gets a clean, runnable version that:
- reads `datasets/Boston1.csv`
- trains a model variant
- saves to `models/<slug>.pkl` (sklearn) or `.keras` (Keras)
"""
from __future__ import annotations
import nbformat as nbf
import pathlib

NB_DIR = pathlib.Path("notebooks")


def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)


def write(path: pathlib.Path, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.13"}}
    nbf.write(nb, str(path))
    print(f"Wrote {path} ({len(cells)} cells)")


# ---------- Linear Regression Boston ----------
write(NB_DIR / "Linear_Regression_Boston.ipynb", [
    md("# Linear Regression — Boston Housing\n\n"
       "Single-feature linear regression. Picks the strongest predictor of `price` and fits a line."),
    md("## Setup"),
    code("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport joblib\nfrom pathlib import Path\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import mean_absolute_error, r2_score"),
    md("## Load data"),
    code("df = pd.read_csv('datasets/Boston1.csv')\ndf.rename(columns={'medv': 'price'}, inplace=True)\ndf.head()"),
    md("## Pick strongest single predictor (lstat)"),
    code("X = df[['lstat']]\ny = df['price']\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"),
    md("## Train"),
    code("model = LinearRegression()\nmodel.fit(X_train, y_train)\nprint(f'Slope:     {model.coef_[0]:.4f}')\nprint(f'Intercept: {model.intercept_:.4f}')"),
    md("## Evaluate"),
    code("y_pred = model.predict(X_test)\nprint(f'MAE: {mean_absolute_error(y_test, y_pred):.4f}')\nprint(f'R²:  {r2_score(y_test, y_pred):.4f}')"),
    md("## Plot fit"),
    code("plt.figure(figsize=(8, 5))\nplt.scatter(X, y, alpha=0.4, label='data')\nx_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)\nplt.plot(x_line, model.predict(x_line), 'r-', linewidth=2, label='fit')\nplt.xlabel('LSTAT (% lower-status pop.)'); plt.ylabel('price (×$1000)')\nplt.title('Linear regression: LSTAT → price'); plt.legend(); plt.grid(alpha=0.2)\nplt.show()"),
    md("## Save"),
    code("MODELS_DIR = Path('models'); MODELS_DIR.mkdir(parents=True, exist_ok=True)\njoblib.dump(model, MODELS_DIR / 'boston_linear.pkl')\nprint('Saved -> models/boston_linear.pkl')"),
])


# ---------- Multiple Regression (Keras) Boston ----------
write(NB_DIR / "Multiple_Regression_Boston.ipynb", [
    md("# Multiple Regression — Boston Housing (Keras)\n\n"
       "Uses **all 13 features** with a small fully-connected neural network."),
    md("## Setup"),
    code("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport joblib\nfrom pathlib import Path\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.metrics import mean_absolute_error, r2_score\nimport tensorflow as tf\nfrom tensorflow.keras.models import Sequential\nfrom tensorflow.keras.layers import Dense, Input"),
    md("## Load data"),
    code("df = pd.read_csv('datasets/Boston1.csv')\ndf.rename(columns={'medv': 'price'}, inplace=True)\nfeatures = [c for c in df.columns if c != 'price']\nX = df[features]\ny = df['price']\nfeatures"),
    md("## Split + scale"),
    code("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\nscaler = StandardScaler()\nX_train_s = scaler.fit_transform(X_train)\nX_test_s  = scaler.transform(X_test)"),
    md("## Build & train"),
    code("model = Sequential([\n    Input(shape=(X_train_s.shape[1],)),\n    Dense(64, activation='relu'),\n    Dense(32, activation='relu'),\n    Dense(1),\n])\nmodel.compile(optimizer='adam', loss='mse', metrics=['mae'])\nhistory = model.fit(X_train_s, y_train, epochs=200, batch_size=16,\n                    validation_split=0.2, verbose=0)\nprint('Final val_mae:', history.history['val_mae'][-1])"),
    md("## Evaluate"),
    code("y_pred = model.predict(X_test_s, verbose=0).flatten()\nprint(f'MAE: {mean_absolute_error(y_test, y_pred):.4f}')\nprint(f'R²:  {r2_score(y_test, y_pred):.4f}')"),
    md("## Save"),
    code("MODELS_DIR = Path('models'); MODELS_DIR.mkdir(parents=True, exist_ok=True)\nmodel.save(MODELS_DIR / 'boston_multi.keras')\njoblib.dump(scaler, MODELS_DIR / 'boston_multi_scaler.pkl')\njoblib.dump(features, MODELS_DIR / 'boston_multi_features.pkl')\nprint('Saved -> models/boston_multi.keras + scaler + feature list')"),
])


# ---------- Multiple Regression Nonlinear Boston ----------
write(NB_DIR / "Multiple_Regression_Boston_Nonlinear.ipynb", [
    md("# Polynomial Regression — Boston Housing\n\n"
       "Adds polynomial features (degree 2) to a linear model."),
    md("## Setup"),
    code("import pandas as pd\nimport numpy as np\nimport joblib\nfrom pathlib import Path\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import PolynomialFeatures, StandardScaler\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.metrics import mean_absolute_error, r2_score"),
    md("## Load + split"),
    code("df = pd.read_csv('datasets/Boston1.csv')\ndf.rename(columns={'medv': 'price'}, inplace=True)\nX = df[['rm', 'lstat', 'ptratio']]\ny = df['price']\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"),
    md("## Pipeline: scale → polynomial → linear"),
    code("model = Pipeline([\n    ('scale', StandardScaler()),\n    ('poly',  PolynomialFeatures(degree=2, include_bias=False)),\n    ('lr',    LinearRegression()),\n])\nmodel.fit(X_train, y_train)"),
    md("## Evaluate"),
    code("y_pred = model.predict(X_test)\nprint(f'MAE: {mean_absolute_error(y_test, y_pred):.4f}')\nprint(f'R²:  {r2_score(y_test, y_pred):.4f}')"),
    md("## Save"),
    code("MODELS_DIR = Path('models'); MODELS_DIR.mkdir(parents=True, exist_ok=True)\njoblib.dump(model, MODELS_DIR / 'boston_poly.pkl')\nprint('Saved -> models/boston_poly.pkl')"),
])


# ---------- Optimized Boston (model comparison) ----------
write(NB_DIR / "Optimized_Boston.ipynb", [
    md("# Optimized Boston — Model Comparison\n\n"
       "Trains three variants on the same RM/LSTAT/PTRATIO features and persists each:\n"
       "- Baseline neural net (no callbacks)\n"
       "- + Early stopping\n"
       "- + Early stopping + extra hidden layer"),
    md("## Setup"),
    code("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport joblib\nfrom pathlib import Path\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\nimport tensorflow as tf\nfrom tensorflow.keras.models import Sequential\nfrom tensorflow.keras.layers import Dense, Input\nfrom tensorflow.keras.callbacks import EarlyStopping"),
    md("## Data"),
    code("df = pd.read_csv('datasets/Boston1.csv')\ndf.rename(columns={'medv': 'price'}, inplace=True)\nX = df[['rm', 'lstat', 'ptratio']]\ny = df['price']\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\nscaler = StandardScaler()\nX_train_s = scaler.fit_transform(X_train)\nX_test_s  = scaler.transform(X_test)"),
    md("## Helpers"),
    code("def evaluate(model, X, y):\n    p = model.predict(X, verbose=0).flatten()\n    return dict(\n        mae=mean_absolute_error(y, p),\n        rmse=float(np.sqrt(mean_squared_error(y, p))),\n        r2=r2_score(y, p),\n    )\n\nearly_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)"),
    md("## Variant 1 — baseline"),
    code("def make_model(extra_layer=False):\n    layers = [Input(shape=(X_train_s.shape[1],)),\n              Dense(32, activation='relu'),\n              Dense(16, activation='relu')]\n    if extra_layer:\n        layers.append(Dense(8, activation='relu'))\n    layers.append(Dense(1))\n    m = Sequential(layers)\n    m.compile(optimizer='adam', loss='mse', metrics=['mae'])\n    return m\n\nbaseline = make_model()\nbaseline.fit(X_train_s, y_train, epochs=300, batch_size=16,\n             validation_split=0.2, verbose=0)\nm_baseline = evaluate(baseline, X_test_s, y_test)\nm_baseline"),
    md("## Variant 2 — early stopping"),
    code("es_model = make_model()\nes_model.fit(X_train_s, y_train, epochs=300, batch_size=16,\n             validation_split=0.2, verbose=0, callbacks=[early_stop])\nm_es = evaluate(es_model, X_test_s, y_test)\nm_es"),
    md("## Variant 3 — early stopping + extra layer"),
    code("es_extra = make_model(extra_layer=True)\nes_extra.fit(X_train_s, y_train, epochs=300, batch_size=16,\n             validation_split=0.2, verbose=0, callbacks=[early_stop])\nm_extra = evaluate(es_extra, X_test_s, y_test)\nm_extra"),
    md("## Comparison"),
    code("results = pd.DataFrame({\n    'baseline':           m_baseline,\n    'early_stop':         m_es,\n    'early_stop_extra':   m_extra,\n}).T\nresults"),
    md("## Save best"),
    code("# Pick model with lowest MAE\nbest_name, best_model = min(\n    [('baseline', baseline), ('early_stop', es_model), ('early_stop_extra', es_extra)],\n    key=lambda kv: evaluate(kv[1], X_test_s, y_test)['mae'],\n)\nMODELS_DIR = Path('models'); MODELS_DIR.mkdir(parents=True, exist_ok=True)\nbest_model.save(MODELS_DIR / 'boston_optimized.keras')\njoblib.dump(scaler, MODELS_DIR / 'boston_optimized_scaler.pkl')\nprint(f'Best variant: {best_name}')\nprint('Saved -> models/boston_optimized.keras')"),
])

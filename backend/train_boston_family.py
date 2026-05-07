"""Train all Boston variants and save artefacts to models/."""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "datasets" / "Boston1.csv")
df.rename(columns={"medv": "price"}, inplace=True)

# ---------- Linear (single feature: lstat) ----------
X = df[["lstat"]]
y = df["price"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
linear = LinearRegression().fit(X_tr, y_tr)
joblib.dump(linear, MODELS / "boston_linear.pkl")
print("Saved boston_linear.pkl")

# ---------- Multiple regression (Keras, all 13 features) ----------
features = [c for c in df.columns if c != "price"]
X = df[features]
y = df["price"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_tr)
X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)
multi = Sequential([Input(shape=(X_tr_s.shape[1],)),
                    Dense(64, activation="relu"),
                    Dense(32, activation="relu"),
                    Dense(1)])
multi.compile(optimizer="adam", loss="mse", metrics=["mae"])
multi.fit(X_tr_s, y_tr, epochs=200, batch_size=16, validation_split=0.2, verbose=0)
multi.save(MODELS / "boston_multi.keras")
joblib.dump(scaler, MODELS / "boston_multi_scaler.pkl")
joblib.dump(features, MODELS / "boston_multi_features.pkl")
print("Saved boston_multi.keras + scaler + features")

# ---------- Polynomial (rm, lstat, ptratio) ----------
X = df[["rm", "lstat", "ptratio"]]
y = df["price"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
poly = Pipeline([
    ("scale", StandardScaler()),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("lr", LinearRegression()),
]).fit(X_tr, y_tr)
joblib.dump(poly, MODELS / "boston_poly.pkl")
print("Saved boston_poly.pkl")

# ---------- Optimized (rm, lstat, ptratio + early stop + extra layer) ----------
scaler_opt = StandardScaler().fit(X_tr)
X_tr_s, X_te_s = scaler_opt.transform(X_tr), scaler_opt.transform(X_te)
opt = Sequential([Input(shape=(X_tr_s.shape[1],)),
                  Dense(32, activation="relu"),
                  Dense(16, activation="relu"),
                  Dense(8, activation="relu"),
                  Dense(1)])
opt.compile(optimizer="adam", loss="mse", metrics=["mae"])
es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
opt.fit(X_tr_s, y_tr, epochs=300, batch_size=16, validation_split=0.2, verbose=0, callbacks=[es])
opt.save(MODELS / "boston_optimized.keras")
joblib.dump(scaler_opt, MODELS / "boston_optimized_scaler.pkl")
print("Saved boston_optimized.keras + scaler")

print("\nAll Boston variants trained.")

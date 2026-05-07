"""Quick-train script: produces backend/models/iris.pkl from sklearn's bundled Iris dataset."""
from __future__ import annotations
import pathlib
import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

OUT = pathlib.Path(__file__).resolve().parent.parent / "models" / "iris.pkl"
OUT.parent.mkdir(parents=True, exist_ok=True)

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = LogisticRegression(max_iter=500)
model.fit(X_tr, y_tr)
print(f"Test accuracy: {model.score(X_te, y_te):.4f}")

joblib.dump(model, OUT)
print(f"Saved -> {OUT}")

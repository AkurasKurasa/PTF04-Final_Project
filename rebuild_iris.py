"""Rebuild Iris.ipynb (MLP) and Iris_Classifier.ipynb (KNN), strip anvil, save both models."""
import nbformat as nbf
import pathlib

def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)

def write(path, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.13"}}
    nbf.write(nb, str(path))
    print(f"Wrote {path}")

# Iris.ipynb — MLP (neural network)
write(pathlib.Path("notebooks/Iris.ipynb"), [
    md("# Iris Classification — Neural Network (MLP)\n\nClassic 4-feature classifier using `sklearn.neural_network.MLPClassifier`."),
    md("## Setup"),
    code("import joblib\nimport numpy as np\nimport matplotlib.pyplot as plt\nfrom pathlib import Path\nfrom sklearn.datasets import load_iris\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.neural_network import MLPClassifier\nfrom sklearn.metrics import accuracy_score, confusion_matrix, classification_report"),
    md("## Load + split"),
    code("iris = load_iris()\nX, y = iris.data, iris.target\nclass_names = list(iris.target_names)\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\nprint('Classes:', class_names)"),
    md("## Train MLP"),
    code("model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42)\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)\nprint(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')\nprint(classification_report(y_test, y_pred, target_names=class_names))"),
    md("## Confusion matrix"),
    code("cm = confusion_matrix(y_test, y_pred)\nfig, ax = plt.subplots(figsize=(5, 4))\nim = ax.imshow(cm, cmap='Blues')\nfor (i, j), v in np.ndenumerate(cm):\n    ax.text(j, i, str(v), ha='center', va='center')\nax.set_xticks(range(3)); ax.set_yticks(range(3))\nax.set_xticklabels(class_names); ax.set_yticklabels(class_names)\nax.set_xlabel('Predicted'); ax.set_ylabel('True')\nplt.colorbar(im); plt.tight_layout(); plt.show()"),
    md("## Save"),
    code("MODELS_DIR = Path('models'); MODELS_DIR.mkdir(parents=True, exist_ok=True)\njoblib.dump(model, MODELS_DIR / 'iris_mlp.pkl')\nprint('Saved -> models/iris_mlp.pkl')"),
])

# Iris_Classifier.ipynb — KNN
write(pathlib.Path("notebooks/Iris_Classifier.ipynb"), [
    md("# Iris Classification — k-Nearest Neighbours\n\nLightweight KNN classifier on the four iris flower features."),
    md("## Setup"),
    code("import joblib\nimport numpy as np\nimport matplotlib.pyplot as plt\nfrom pathlib import Path\nfrom sklearn.datasets import load_iris\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.metrics import accuracy_score, classification_report, confusion_matrix"),
    md("## Load + split"),
    code("iris = load_iris()\nX, y = iris.data, iris.target\nclass_names = list(iris.target_names)\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)"),
    md("## Try several k values"),
    code("scores = {}\nfor k in range(1, 11):\n    m = KNeighborsClassifier(n_neighbors=k)\n    m.fit(X_train, y_train)\n    scores[k] = accuracy_score(y_test, m.predict(X_test))\n\nfor k, s in scores.items():\n    print(f'k={k:2d} -> {s:.4f}')"),
    md("## Pick best k, train final"),
    code("best_k = max(scores, key=scores.get)\nprint('Best k:', best_k)\nmodel = KNeighborsClassifier(n_neighbors=best_k)\nmodel.fit(X_train, y_train)\nprint(classification_report(y_test, model.predict(X_test), target_names=class_names))"),
    md("## Save"),
    code("MODELS_DIR = Path('models'); MODELS_DIR.mkdir(parents=True, exist_ok=True)\njoblib.dump(model, MODELS_DIR / 'iris_knn.pkl')\nprint('Saved -> models/iris_knn.pkl')"),
])

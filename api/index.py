"""Vercel serverless entrypoint for the Flask backend."""
from __future__ import annotations
import os
import sys
import traceback
import pathlib

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

ROOT = pathlib.Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from flask import Flask, jsonify

# Try to import the real backend Flask app. If anything goes wrong, expose a
# tiny diagnostic Flask app instead of letting the function crash with a
# generic 500.
try:
    from app import app          # noqa: E402  (the real Flask WSGI handler)
    _import_error: str | None = None
except Exception as e:           # noqa: BLE001
    _import_error = "".join(traceback.format_exception(e))
    app = Flask(__name__)

    @app.get("/api/health")
    def _health():
        return jsonify({
            "status": "import_failed",
            "error": _import_error,
            "root_listing": sorted(p.name for p in ROOT.iterdir())[:30],
            "backend_listing": (
                sorted(p.name for p in BACKEND.iterdir())[:30]
                if BACKEND.exists() else "missing"
            ),
            "models_listing": (
                sorted(p.name for p in (ROOT / "models").iterdir())[:30]
                if (ROOT / "models").exists() else "missing"
            ),
            "sys_path_head": sys.path[:5],
        })

    @app.route("/", defaults={"_p": ""})
    @app.route("/<path:_p>")
    def _catchall(_p):
        return jsonify({"error": "backend import failed", "detail": _import_error}), 500


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


# Vercel's @vercel/python builder picks up `app` (Flask WSGI handler).
application = app

"""Vercel serverless entrypoint. Re-exposes the Flask backend WSGI app."""
from __future__ import annotations
import os
import sys
import traceback
import pathlib

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from flask import Flask, jsonify

try:
    from app import app  # noqa: E402   (Flask WSGI app from backend/app.py)
    _import_error: str | None = None
except Exception as e:    # noqa: BLE001
    _import_error = "".join(traceback.format_exception(e))
    app = Flask(__name__)

    @app.route("/", defaults={"_p": ""})
    @app.route("/<path:_p>")
    def _failed(_p):
        return jsonify({"error": "backend import failed", "detail": _import_error}), 500


# Vercel's @vercel/python builder picks up `app` (Flask WSGI handler).
application = app

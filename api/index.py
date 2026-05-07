"""Vercel serverless entrypoint. Imports the Flask app from /backend/app.py
and re-exposes it under the name `app` so Vercel's Python builder picks it up."""
import os
import sys
import pathlib

# Make sibling /backend importable
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

# Skip writing .pyc on Vercel build/runtime
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

from flask import Flask  # noqa: F401  (signal to Vercel: this is a Flask serverless fn)
from app import app      # noqa: E402  (Flask WSGI app, served by @vercel/python)

# Expose under both common names
application = app

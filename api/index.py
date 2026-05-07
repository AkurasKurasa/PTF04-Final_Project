"""Vercel serverless entrypoint. Imports the Flask app from /backend/app.py."""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app import app  # noqa: E402,F401   (app is the Flask WSGI handler)

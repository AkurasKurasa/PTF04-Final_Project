"""Bare-bones Vercel diagnostic. If this 500s the runtime itself is broken."""
from http.server import BaseHTTPRequestHandler
import json
import os
import sys
import pathlib


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        info = {
            "ok": True,
            "python": sys.version,
            "cwd": os.getcwd(),
            "file": __file__,
            "sys_path": sys.path[:6],
            "env_vercel": os.environ.get("VERCEL", ""),
            "env_region": os.environ.get("VERCEL_REGION", ""),
        }
        try:
            here = pathlib.Path(__file__).resolve().parent.parent
            info["root"] = str(here)
            info["root_listing"] = sorted([p.name for p in here.iterdir()])[:40]
            backend = here / "backend"
            info["backend_exists"] = backend.exists()
            if backend.exists():
                info["backend_listing"] = sorted([p.name for p in backend.iterdir()])[:40]
            models = here / "models"
            info["models_exists"] = models.exists()
            if models.exists():
                info["models_listing"] = sorted([p.name for p in models.iterdir()])[:40]
        except Exception as e:
            info["fs_error"] = repr(e)

        for mod in ("flask", "numpy", "joblib", "sklearn", "onnxruntime", "PIL"):
            try:
                __import__(mod)
                info[f"import_{mod}"] = "ok"
            except Exception as e:
                info[f"import_{mod}"] = f"FAIL: {e!r}"

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(info, indent=2).encode("utf-8"))

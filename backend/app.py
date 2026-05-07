from __future__ import annotations
import importlib
import json
import pathlib
import traceback

import queue
import threading

from flask import Flask, jsonify, request, send_from_directory, abort, Response, stream_with_context

ROOT = pathlib.Path(__file__).resolve().parent.parent
BACKEND = pathlib.Path(__file__).resolve().parent
SCHEMAS = BACKEND / "schemas"

app = Flask(__name__)


# ---------- Kernel pool: one persistent kernel per slug ----------
_kernels: dict[str, object] = {}
_kernel_lock = threading.Lock()


def _get_kernel(slug: str):
    from jupyter_client.manager import KernelManager
    with _kernel_lock:
        km = _kernels.get(slug)
        if km is None or not km.is_alive():
            km = KernelManager(kernel_name="python3")
            km.start_kernel(cwd=str(ROOT))
            _kernels[slug] = km
        return km


def _shutdown_kernel(slug: str):
    with _kernel_lock:
        km = _kernels.pop(slug, None)
    if km is not None:
        try:
            km.shutdown_kernel(now=True)
        except Exception:
            pass


def _run_code(km, code: str, timeout: int = 1800) -> list[dict]:
    kc = km.client()
    kc.start_channels()
    try:
        msg_id = kc.execute(code, store_history=True)
        outputs: list[dict] = []
        while True:
            try:
                msg = kc.get_iopub_msg(timeout=timeout)
            except queue.Empty:
                outputs.append({"type": "error", "ename": "Timeout",
                                "evalue": f"No output for {timeout}s",
                                "traceback": []})
                break
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            mtype = msg["msg_type"]
            content = msg.get("content", {})
            if mtype == "status" and content.get("execution_state") == "idle":
                break
            if mtype == "stream":
                outputs.append({"type": "stream", "name": content.get("name"),
                                "text": content.get("text", "")})
            elif mtype in ("execute_result", "display_data"):
                outputs.append({"type": mtype, "data": content.get("data", {})})
            elif mtype == "error":
                outputs.append({"type": "error",
                                "ename": content.get("ename", ""),
                                "evalue": content.get("evalue", ""),
                                "traceback": content.get("traceback", [])})
        return outputs
    finally:
        kc.stop_channels()


def _slug(name: str) -> str:
    return name.replace(".html", "").lower()


@app.get("/")
def index():
    return send_from_directory(str(ROOT), "index.html")


@app.post("/api/run/<slug>")
def run_notebook(slug: str):
    import subprocess
    slug = _slug(slug)
    nb_dir = ROOT / "notebooks"
    pages_dir = ROOT / "pages"
    matches = [p for p in nb_dir.glob("*.ipynb") if p.stem.lower() == slug]
    if not matches:
        return jsonify({"error": f"Notebook {slug} not found."}), 404
    nb = matches[0]
    try:
        cmd = [
            "jupyter", "nbconvert",
            "--to", "html",
            "--template", "basic",
            "--execute",
            "--allow-errors",
            "--ExecutePreprocessor.timeout=1800",
            "--ExecutePreprocessor.allow_errors=True",
            f"--ExecutePreprocessor.cwd={ROOT}",
            "--output-dir", str(pages_dir),
            str(nb),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900, encoding="utf-8", errors="replace", cwd=str(ROOT))
        out_html = pages_dir / f"{nb.stem}.html"
        if proc.returncode != 0 and not out_html.exists():
            return jsonify({"error": "Execution failed.", "stderr": proc.stderr[-2000:]}), 500
        # Count cells with errors by scanning output html
        errors = 0
        if out_html.exists():
            txt = out_html.read_text(encoding="utf-8", errors="replace")
            errors = txt.count("output_error") + txt.lower().count("traceback")
        return jsonify({
            "ok": True,
            "file": out_html.name,
            "errors": errors,
            "stderr": proc.stderr[-2000:] if proc.stderr else "",
        })
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Execution timed out (>15 min)."}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/api/<slug>/schema")
def schema(slug: str):
    slug = _slug(slug)
    path = SCHEMAS / f"{slug}.json"
    if not path.exists():
        return jsonify({"error": "Demo not yet configured for this notebook."}), 404
    return jsonify(json.loads(path.read_text(encoding="utf-8")))


@app.post("/api/<slug>/predict")
def predict(slug: str):
    slug = _slug(slug)
    try:
        module = importlib.import_module(f"predictors.{slug}")
    except ModuleNotFoundError:
        return jsonify({"error": "Predictor module missing."}), 404

    data = {}
    if request.is_json:
        data = request.get_json(silent=True) or {}
    else:
        data = request.form.to_dict(flat=True)

    files = request.files

    try:
        result = module.predict(data, files)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.get("/api/status/<slug>")
def status(slug: str):
    """Report execution status: last run timestamp, whether HTML exists."""
    slug = _slug(slug)
    nb_dir = ROOT / "notebooks"
    pages_dir = ROOT / "pages"
    matches = [p for p in nb_dir.glob("*.ipynb") if p.stem.lower() == slug]
    if not matches:
        return jsonify({"error": "not found"}), 404
    nb = matches[0]
    html = pages_dir / f"{nb.stem}.html"
    snapshot = pages_dir / f"{nb.stem}.executed.ipynb"
    return jsonify({
        "notebook_mtime": nb.stat().st_mtime,
        "html_mtime": html.stat().st_mtime if html.exists() else None,
        "snapshot_mtime": snapshot.stat().st_mtime if snapshot.exists() else None,
        "executed": snapshot.exists(),
        "stale": html.exists() and html.stat().st_mtime < nb.stat().st_mtime,
    })


@app.post("/api/refresh/<slug>")
def refresh_notebook(slug: str):
    """Regenerate HTML from .ipynb without executing cells (fast). Prefers executed snapshot."""
    import subprocess
    slug = _slug(slug)
    nb_dir = ROOT / "notebooks"
    pages_dir = ROOT / "pages"
    matches = [p for p in nb_dir.glob("*.ipynb") if p.stem.lower() == slug]
    if not matches:
        return jsonify({"error": "not found"}), 404
    nb = matches[0]
    out_html = pages_dir / f"{nb.stem}.html"
    snapshot = pages_dir / f"{nb.stem}.executed.ipynb"

    # If snapshot is newer than source, regenerate from snapshot to keep outputs
    source = nb
    if snapshot.exists() and snapshot.stat().st_mtime >= nb.stat().st_mtime:
        source = snapshot

    if out_html.exists() and out_html.stat().st_mtime >= source.stat().st_mtime:
        return jsonify({"ok": True, "file": out_html.name, "regenerated": False})

    try:
        cmd = [
            "jupyter", "nbconvert",
            "--to", "html",
            "--template", "basic",
            "--output-dir", str(pages_dir),
            str(source),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                              encoding="utf-8", errors="replace", cwd=str(ROOT))
        if proc.returncode != 0 and not out_html.exists():
            return jsonify({"error": proc.stderr[-1000:]}), 500
        # Rename snapshot's HTML output to match base stem
        if source != nb:
            generated = pages_dir / f"{source.stem}.html"
            if generated.exists() and generated != out_html:
                out_html.unlink(missing_ok=True)
                generated.rename(out_html)
        return jsonify({"ok": True, "file": out_html.name, "regenerated": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/api/cell/<slug>")
def run_cell(slug: str):
    # Author-only: viewers can't execute arbitrary code on the kernel
    if request.headers.get("X-Author-Mode") != "1":
        return jsonify({"error": "Cell execution disabled in viewer mode."}), 403
    slug = _slug(slug)
    body = request.get_json(silent=True) or {}
    code = body.get("code", "")
    if not code.strip():
        return jsonify({"outputs": []})
    try:
        km = _get_kernel(slug)
        outputs = _run_code(km, code)
        return jsonify({"outputs": outputs})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.post("/api/kernel/<slug>/restart")
def restart_kernel(slug: str):
    slug = _slug(slug)
    _shutdown_kernel(slug)
    return jsonify({"ok": True})


@app.get("/api/run-stream/<slug>")
def run_stream(slug: str):
    slug = _slug(slug)
    nb_dir = ROOT / "notebooks"
    pages_dir = ROOT / "pages"
    matches = [p for p in nb_dir.glob("*.ipynb") if p.stem.lower() == slug]
    if not matches:
        return jsonify({"error": "not found"}), 404
    nb_path = matches[0]

    def sse(obj):
        return f"data: {json.dumps(obj)}\n\n"

    def gen():
        import nbformat
        from nbclient import NotebookClient
        from nbconvert import HTMLExporter

        try:
            nb = nbformat.read(str(nb_path), as_version=4)
        except Exception as e:
            yield sse({"event": "error", "msg": f"Cannot read notebook: {e}"})
            return

        total_cells = len(nb.cells)
        code_total = sum(1 for c in nb.cells if c.cell_type == "code")
        yield sse({"event": "start", "total": total_cells, "code_total": code_total})

        client = NotebookClient(
            nb,
            timeout=1800,
            allow_errors=True,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        )

        try:
            client.create_kernel_manager()
            client.start_new_kernel()
            client.start_new_kernel_client()
        except Exception as e:
            yield sse({"event": "error", "msg": f"Kernel start failed: {e}"})
            return

        code_idx = 0
        try:
            for i, cell in enumerate(nb.cells):
                if cell.cell_type != "code":
                    yield sse({"event": "cell-skip", "index": i})
                    continue
                code_idx += 1
                yield sse({"event": "cell-start", "index": i, "code_idx": code_idx})
                try:
                    client.execute_cell(cell, i)
                    has_err = any(
                        o.get("output_type") == "error"
                        for o in cell.get("outputs", [])
                    )
                    yield sse({
                        "event": "cell-done",
                        "index": i,
                        "code_idx": code_idx,
                        "error": has_err,
                    })
                except Exception as e:
                    yield sse({
                        "event": "cell-done",
                        "index": i,
                        "code_idx": code_idx,
                        "error": True,
                        "msg": str(e),
                    })
        finally:
            try:
                client._cleanup_kernel()
            except Exception:
                pass

        try:
            exporter = HTMLExporter(template_name="basic")
            html, _ = exporter.from_notebook_node(nb)
            out = pages_dir / f"{nb_path.stem}.html"
            out.write_text(html, encoding="utf-8")
            # Save executed snapshot so reload doesn't strip outputs
            snapshot = pages_dir / f"{nb_path.stem}.executed.ipynb"
            nbformat.write(nb, str(snapshot))
            yield sse({"event": "done", "file": out.name})
        except Exception as e:
            yield sse({"event": "error", "msg": f"Render failed: {e}"})

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }
    return Response(stream_with_context(gen()), headers=headers)


@app.get("/<path:filename>")
def assets(filename: str):
    target = ROOT / filename
    if target.exists() and target.is_file():
        return send_from_directory(str(ROOT), filename)
    abort(404)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)

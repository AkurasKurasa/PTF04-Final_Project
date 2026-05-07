"""Convert all .keras → .onnx via tf2onnx CLI (Keras 3 compatible)."""
from __future__ import annotations
import pathlib
import subprocess
import sys

MODELS = pathlib.Path(__file__).resolve().parent.parent / "models"

SOURCES = [
    "boston_house_price.keras",
    "boston_multi.keras",
    "boston_optimized.keras",
    "cifar10_cnn.keras",
    "fashion_mnist.keras",
    "dogs_vs_cats.keras",
    "sentiment_lstm.keras",
]

failures = []
for name in SOURCES:
    src = MODELS / name
    dst = src.with_suffix(".onnx")
    if not src.exists():
        print(f"SKIP   {name} (missing)")
        continue
    print(f"[{name}] -> {dst.name}")
    proc = subprocess.run(
        [sys.executable, "-m", "tf2onnx.convert",
         "--keras", str(src),
         "--output", str(dst),
         "--opset", "15"],
        capture_output=True, text=True,
    )
    if proc.returncode == 0 and dst.exists():
        print(f"  OK   ({dst.stat().st_size/1024:.1f} KB)")
    else:
        failures.append(name)
        print(f"  FAIL")
        print("  ", (proc.stderr or proc.stdout).splitlines()[-3:])

print("\nFailures:", failures or "none")

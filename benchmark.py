"""Latency / FPS benchmark of the ONNX model on the current device (run it on the board)."""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from detector import OnnxDetector

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/best.onnx")
ap.add_argument("--image", default=None)
ap.add_argument("--runs", type=int, default=50)
args = ap.parse_args()

img_path = args.image or str(next(Path("data/vehicles/images/val").glob("*.jpg")))
img = cv2.imread(img_path)
det = OnnxDetector(args.model)
for _ in range(5):
    det(img)
t = []
for _ in range(args.runs):
    s = time.perf_counter(); det(img); t.append((time.perf_counter() - s) * 1000)
t = np.array(t)
print(f"providers: {det.providers}")
print(f"model size: {Path(args.model).stat().st_size / 1e6:.1f} MB")
print(f"latency mean {t.mean():.1f} ms | p95 {np.percentile(t, 95):.1f} ms | {1000 / t.mean():.1f} FPS")

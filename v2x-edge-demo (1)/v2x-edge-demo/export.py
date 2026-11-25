"""Export the fine-tuned model to ONNX (portable: laptop CPU, Jetson, Raspberry Pi...)."""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="models/best.pt")
ap.add_argument("--imgsz", type=int, default=640)
args = ap.parse_args()

out = YOLO(args.weights).export(format="onnx", imgsz=args.imgsz, opset=12, dynamic=False)
target = Path("models/best.onnx")
if Path(out).resolve() != target.resolve():
    shutil.copy(out, target)
print(f"ONNX model: {target} ({target.stat().st_size / 1e6:.1f} MB)")

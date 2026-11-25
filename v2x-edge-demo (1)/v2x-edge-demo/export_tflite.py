"""Export TFLite/LiteRT variants with the NEW Ultralytics API (>= 8.4: format='litert').

Produces (in the current folder):  <name>.tflite (FP32), <name>_w8a32.tflite (int8 weights),
<name>_int8.tflite (static int8, needed for the NPU). Input layout is NCHW (handled by tflite_utils.py).
  python export_tflite.py --weights yolov8n.pt --imgsz 320 --data coco128.yaml
Colab only (LiteRT export runs on Linux x86_64 / macOS).
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="models/best.pt")
ap.add_argument("--data", default="data/opv2v.yaml", help="calibration data for static INT8 (>300 images recommended)")
ap.add_argument("--imgsz", type=int, default=320)
ap.add_argument("--fp32-only", action="store_true", help="skip the INT8 variants")
args = ap.parse_args()

m = YOLO(args.weights)
m.export(format="litert", imgsz=args.imgsz)                                     # FP32
if not args.fp32_only:
    m.export(format="litert", imgsz=args.imgsz, quantize="w8a32")               # dynamic INT8 weights
    m.export(format="litert", imgsz=args.imgsz, quantize=8, data=args.data)     # static INT8 (needs calibration data)
stem = Path(args.weights).with_suffix("")
Path("models").mkdir(exist_ok=True)
for f in sorted(stem.parent.glob(stem.name + "*.tflite")):
    shutil.copy(f, Path("models") / f.name)
    print(f, f"{f.stat().st_size / 1e6:.2f} MB")

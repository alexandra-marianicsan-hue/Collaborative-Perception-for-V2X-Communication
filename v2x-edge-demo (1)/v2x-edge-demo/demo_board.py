"""Demo on the i.MX 8M Plus (FP32 TFLite, CPU) - collaborative perception, 2 simulated vehicles.

Needs only: numpy, tflite_runtime, pillow (pip3 install pillow).
  python3 demo_board.py --model models/yolov8n.tflite --images demo_images
Each image = one road scene. Vehicle A sees the left part, vehicle B the right part (20% overlap).
Each runs the detector; only detections are 'sent' and fused. Annotated results -> results/demo/.
"""
import argparse
import io
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from collab import det_msg_bytes, fuse, shift
from tflite_utils import TfliteDetector

COCO_VEH = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CUSTOM_VEH = {0: "bicycle", 1: "car", 2: "motorcycle", 3: "bus", 4: "truck"}   # our fine-tuned 5-class model
VEH = COCO_VEH


def letterbox(img, S):
    w, h = img.size
    r = min(S / w, S / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    canvas = Image.new("RGB", (S, S), (114, 114, 114))
    left, top = (S - nw) // 2, (S - nh) // 2
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (left, top))
    return np.asarray(canvas), r, left, top


def detect(det, view, conf):
    arr, r, left, top = letterbox(view, det.imgsz)
    t = time.perf_counter()
    out, _, _ = det.run(arr)
    ms = (time.perf_counter() - t) * 1000
    d = det.decode(out, conf)
    d = d[np.isin(d[:, 5], list(VEH))] if len(d) else d
    if len(d):
        d[:, [0, 2]] = ((d[:, [0, 2]] - left) / r).clip(0, view.size[0])
        d[:, [1, 3]] = ((d[:, [1, 3]] - top) / r).clip(0, view.size[1])
    return d, ms


def draw(img, dets, color):
    img = img.copy(); dr = ImageDraw.Draw(img)
    for x1, y1, x2, y2, c, k in dets:
        dr.rectangle([x1, y1, x2, y2], outline=color, width=3)
        dr.text((x1 + 2, max(0, y1 - 12)), f"{VEH[int(k)]} {c:.2f}", fill=color)
    return img


def jpeg_size(img):
    b = io.BytesIO(); img.save(b, "JPEG", quality=80); return b.tell()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/yolov8n.tflite")
    ap.add_argument("--images", default="demo_images")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--delegate", default=None, help="e.g. /usr/lib/libvx_delegate.so (NPU, int8 models only)")
    args = ap.parse_args()

    det = TfliteDetector(args.model, args.delegate, args.threads)
    global VEH
    shp = det.it.get_output_details()[0]["shape"]                   # (1, 4+nc, N)
    nc = int(min(shp[1:])) - 4
    VEH = CUSTOM_VEH if nc == 5 else COCO_VEH
    print(f"Detected {nc} classes -> {'fine-tuned vehicle model' if nc == 5 else 'COCO pretrained model'}")
    det.run(np.zeros((det.imgsz, det.imgsz, 3), np.uint8))          # warm-up
    out_dir = Path("results/demo"); out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    print(f"Model {args.model} | {"NPU (VX delegate)" if args.delegate else f"CPU x{args.threads} (XNNPACK)"} | input {det.imgsz}x{det.imgsz} | {len(files)} scenes\n")
    tot_ms, tot_jpg, tot_msg, n = 0.0, 0, 0, 0
    for p in files:
        img = Image.open(p).convert("RGB"); W, H = img.size
        a_end, b_start = int(W * (0.5 + args.overlap / 2)), int(W * (0.5 - args.overlap / 2))
        va, vb = img.crop((0, 0, a_end, H)), img.crop((b_start, 0, W, H))
        da, ma = detect(det, va, args.conf)
        db, mb = detect(det, vb, args.conf)
        fused = fuse([shift(da, (0, 0)), shift(db, (b_start, 0))], 0.8)
        jpg, msg = jpeg_size(va) + jpeg_size(vb), det_msg_bytes(da) + det_msg_bytes(db)
        tot_ms += (ma + mb) / 2; tot_jpg += jpg; tot_msg += msg; n += 1
        print(f"{p.name}: A={len(da)} B={len(db)} fused={len(fused)} | {(ma + mb) / 2:.0f} ms/vehicle "
              f"({1000 / ((ma + mb) / 2):.1f} FPS) | uplink JPEG {jpg / 1e3:.1f} kB vs detections {msg} B")
        canvas = Image.new("RGB", (W, H * 2 + 0), (0, 0, 0))
        canvas.paste(draw(va, da, (255, 140, 0)), (0, 0)); canvas.paste(draw(vb, db, (0, 140, 255)), (b_start, 0))
        canvas.paste(draw(img, fused, (0, 200, 0)), (0, H))
        canvas.save(out_dir / f"{p.stem}_demo.jpg")
    if n:
        print(f"\nSUMMARY: {tot_ms / n:.0f} ms avg per vehicle inference | uplink per scene: "
              f"JPEG {tot_jpg / n / 1e3:.1f} kB vs detections {tot_msg / n:.0f} B ({tot_jpg / max(tot_msg, 1):.0f}x less)")
        print(f"Images saved in {out_dir}/ (top: vehicle A orange + vehicle B blue, bottom: fused green)")


if __name__ == "__main__":
    main()

"""Quick pack WITHOUT OPV2V: builds data/pack from the COCO128 vehicle subset (prepare_data.py).
Lets bench_tflite.py measure latency + accuracy today. Boxes are class-agnostic 'vehicle'."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="data/vehicles")
ap.add_argument("--split", default="val")
ap.add_argument("--imgsz", type=int, default=320)
ap.add_argument("--out", default="data/pack")
args = ap.parse_args()

S = args.imgsz
images, metas = [], []
for p in sorted((Path(args.src) / "images" / args.split).glob("*.jpg")):
    img = cv2.imread(str(p)); H, W = img.shape[:2]
    r = min(S / H, S / W); nh, nw = int(round(H * r)), int(round(W * r))
    top, left = (S - nh) // 2, (S - nw) // 2
    canvas = np.full((S, S, 3), 114, np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh))
    gt = []
    lp = Path(args.src) / "labels" / args.split / (p.stem + ".txt")
    for line in lp.read_text().split("\n") if lp.exists() else []:
        if line.strip():
            _, cx, cy, w, h = map(float, line.split()[:5])
            gt.append([(cx - w / 2) * W * r + left, (cy - h / 2) * H * r + top,
                       (cx + w / 2) * W * r + left, (cy + h / 2) * H * r + top])
    images.append(canvas[:, :, ::-1]); metas.append(dict(name=p.name, gt=gt))
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out / "pack.npz", images=np.stack(images))
json.dump(dict(imgsz=S, images=metas, timestamps=[]), open(out / "pack.json", "w"))
print(f"{len(images)} images -> {out}")

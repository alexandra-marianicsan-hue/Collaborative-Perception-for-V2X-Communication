"""Pack the val images (letterboxed, uint8) + metadata into data/pack/ so the BOARD scripts
need only numpy + tflite_runtime (no OpenCV, no PyTorch, no OPV2V download on the board)."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--index", default="data/opv2v_index.json")
ap.add_argument("--imgsz", type=int, default=320)
ap.add_argument("--out", default="data/pack")
args = ap.parse_args()

S = args.imgsz
idx = json.load(open(args.index))
root = Path(idx["root"])
images, metas, tss = [], [], []
for t in idx["timestamps"]:
    ids = []
    for c in t["cavs"]:
        img = cv2.imread(str(root / c["image"]))
        H, W = img.shape[:2]
        r = min(S / H, S / W)
        nh, nw = int(round(H * r)), int(round(W * r))
        top, left = (S - nh) // 2, (S - nw) // 2
        canvas = np.full((S, S, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh))
        images.append(canvas[:, :, ::-1])                                  # RGB
        gt = [[b[0] * r + left, b[1] * r + top, b[2] * r + left, b[3] * r + top] for b in c["gt2d"]]
        ids.append(len(metas))
        metas.append(dict(scenario=t["scenario"], ts=t["ts"], cav=c["cav"], ego=c["cav"] == t["ego"],
                          pose=c["pose"], K=c["K"], r=r, left=left, top=top, W=W, H=H, gt=gt))
    tss.append(dict(scenario=t["scenario"], ts=t["ts"], ego=t["ego"], gt_world=t["gt_world"], image_ids=ids))
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out / "pack.npz", images=np.stack(images))
json.dump(dict(imgsz=S, ground_offset=idx["ground_offset"], images=metas, timestamps=tss), open(out / "pack.json", "w"))
print(f"{len(images)} images, {len(tss)} timestamps -> {out}  ({(out / 'pack.npz').stat().st_size / 1e6:.1f} MB)")

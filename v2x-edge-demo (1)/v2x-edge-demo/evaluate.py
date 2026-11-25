"""Evaluation metrics for the report/demo.

1) Detector quality (Ultralytics val on the ONNX model): mAP50, mAP50-95.
2) Collaborative perception: precision / recall / F1 against ground truth for
   - full image (reference, one device sees everything)
   - vehicle A alone, vehicle B alone (partial views)
   - fused A+B (collaborative)
"""
import argparse
from pathlib import Path

import cv2

from collab import fuse, load_gt, make_views, match_counts, shift
from detector import OnnxDetector

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def prf(tp, fp, fn):
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/best.onnx")
    ap.add_argument("--images", default="data/vehicles/images/val")
    ap.add_argument("--labels", default="data/vehicles/labels/val")
    ap.add_argument("--data", default="data/vehicles.yaml")
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--map", action="store_true", help="also run Ultralytics val for mAP")
    args = ap.parse_args()

    lines = ["# Evaluation", ""]
    if args.map:
        try:
            from ultralytics import YOLO
            m = YOLO(args.model, task="detect").val(data=args.data, imgsz=640, split="val", verbose=False)
            lines += [f"Detector (ONNX, val set): mAP50 = {m.box.map50:.3f}, mAP50-95 = {m.box.map:.3f}", ""]
        except Exception as e:
            lines += [f"(mAP skipped: {e})", ""]

    det = OnnxDetector(args.model, conf=args.conf)
    tot = {k: [0, 0, 0] for k in ("full image", "vehicle A only", "vehicle B only", "fused A+B")}
    imgs = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in IMG_EXT)
    for p in imgs:
        img = cv2.imread(str(p))
        H, W = img.shape[:2]
        gt = load_gt(Path(args.labels) / (p.stem + ".txt"), W, H)
        (va, oa), (vb, ob) = make_views(img, args.overlap)
        da, db = shift(det(va), oa), shift(det(vb), ob)
        res = {"full image": det(img), "vehicle A only": da, "vehicle B only": db, "fused A+B": fuse([da, db])}
        for k, d in res.items():
            for i, v in enumerate(match_counts(d, gt)):
                tot[k][i] += v

    lines += [f"Images: {len(imgs)} | IoU>=0.5 | overlap between views: {args.overlap:.0%}", "",
              "| setup | precision | recall | F1 |", "|---|---|---|---|"]
    for k, (tp, fp, fn) in tot.items():
        p, r, f = prf(tp, fp, fn)
        lines.append(f"| {k} | {p:.3f} | {r:.3f} | {f:.3f} |")
    text = "\n".join(lines)
    print(text)
    Path("results").mkdir(exist_ok=True)
    Path("results/evaluation.md").write_text(text + "\n")


if __name__ == "__main__":
    main()

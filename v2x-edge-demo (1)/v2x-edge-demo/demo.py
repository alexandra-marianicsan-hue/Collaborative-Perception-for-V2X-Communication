"""Live demo on the edge device: two simulated vehicles + late fusion.

  python demo.py --source data/vehicles/images/val          # image folder
  python demo.py --source video.mp4                         # video file
  python demo.py --source 0                                 # webcam
  python demo.py --source data/vehicles/images/val --no-show --save   # headless board

Each frame is split into two overlapping views (vehicle A / vehicle B). Each view is
detected independently (ONNX Runtime), then only the detections are 'sent' and fused.
Overlay shows latency/FPS and bandwidth: raw view vs JPEG vs detection message.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from collab import det_msg_bytes, fuse, jpeg_bytes, make_views, shift
from detector import OnnxDetector

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
COL_A, COL_B, COL_F = (255, 140, 0), (0, 140, 255), (0, 200, 0)


def frames(source):
    if source.isdigit() or Path(source).is_file():
        cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        while True:
            ok, f = cap.read()
            if not ok:
                break
            yield f
        cap.release()
    else:
        for p in sorted(Path(source).iterdir()):
            if p.suffix.lower() in IMG_EXT:
                yield cv2.imread(str(p))


def draw(img, dets, names, color):
    out = img.copy()
    for x1, y1, x2, y2, c, k in dets:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(out, f"{names.get(int(k), int(k))} {c:.2f}", (int(x1), max(12, int(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def label(img, text, color):
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/best.onnx")
    ap.add_argument("--source", default="data/vehicles/images/val")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--fuse-thr", type=float, default=0.8)
    ap.add_argument("--delay", type=int, default=1500, help="ms per image (folder mode)")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--save", action="store_true", help="save annotated frames to results/demo/")
    args = ap.parse_args()

    det = OnnxDetector(args.model, conf=args.conf)
    print(f"Model: {args.model} | providers: {det.providers} | classes: {det.names}")
    is_folder = Path(args.source).is_dir()
    out_dir = Path("results/demo")
    if args.save:
        out_dir.mkdir(parents=True, exist_ok=True)
    show = not args.no_show
    log = []

    for n, frame in enumerate(frames(args.source)):
        views = make_views(frame, args.overlap)
        dets, lat = [], []
        for v, _ in views:
            t = time.perf_counter()
            dets.append(det(v))
            lat.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        fused = fuse([shift(d, off) for d, (_, off) in zip(dets, views)], args.fuse_thr)
        t_fuse = (time.perf_counter() - t) * 1000

        raw = sum(v.nbytes for v, _ in views)
        jpg = sum(jpeg_bytes(v) for v, _ in views)
        msg = sum(det_msg_bytes(d) for d in dets)
        ms = float(np.mean(lat))
        log.append(dict(frame=n, lat_ms=ms, fuse_ms=t_fuse, n_A=len(dets[0]), n_B=len(dets[1]),
                        n_fused=len(fused), raw_B=raw, jpg_B=jpg, msg_B=msg))

        pa = draw(views[0][0], dets[0], det.names, COL_A)
        pb = draw(views[1][0], dets[1], det.names, COL_B)
        label(pa, f"Vehicle A  ({len(dets[0])} det)", COL_A)
        label(pb, f"Vehicle B  ({len(dets[1])} det)", COL_B)
        top = np.hstack([pa, pb])
        fp = draw(frame, fused, det.names, COL_F)
        label(fp, f"Fused (collaborative): {len(fused)} det", COL_F)
        lines = [f"edge inference/vehicle: {ms:.1f} ms ({1000 / ms:.1f} FPS) | fusion {t_fuse:.2f} ms",
                 f"uplink A+B: raw {raw / 1e3:.0f} kB | JPEG {jpg / 1e3:.1f} kB | detections {msg} B "
                 f"(x{jpg / max(msg, 1):.0f} smaller than JPEG)"]
        for i, s in enumerate(lines):
            cv2.putText(fp, s, (6, fp.shape[0] - 28 + 14 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
            
        if args.save:
            w = 1100
            stack = np.vstack([cv2.resize(top, (w, int(top.shape[0] * w / top.shape[1]))),
                               cv2.resize(fp, (w, int(fp.shape[0] * w / fp.shape[1])))])
            cv2.imwrite(str(out_dir / f"frame_{n:04d}.jpg"), stack)
        if show:
            try:
                cv2.imshow("Vehicles A | B", top)
                cv2.imshow("Fused perception", fp)
                if (cv2.waitKey(args.delay if is_folder else 1) & 0xFF) in (ord("q"), 27):
                    break
            except cv2.error:
                print("No display available -> use --no-show --save"); show = False
        print(f"[{n:03d}] A={len(dets[0])} B={len(dets[1])} fused={len(fused)} "
              f"lat={ms:.1f} ms  uplink: JPEG {jpg} B vs det {msg} B")

    if log:
        s = {k: float(np.mean([r[k] for r in log])) for k in log[0] if k != "frame"}
        s["frames"] = len(log)
        Path("results").mkdir(exist_ok=True)
        json.dump(s, open("results/demo_summary.json", "w"), indent=2)
        print("\nSummary:", json.dumps(s, indent=2))
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

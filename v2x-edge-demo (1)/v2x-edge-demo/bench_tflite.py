"""Latency + accuracy benchmark of one TFLite model on one backend (CPU or NPU).

Run the SAME script on the board for each configuration; each run appends a row to
results/benchmark.md (this is the before/after + NPU-vs-CPU table for the report).

  # CPU (4 threads, XNNPACK)
  python3 bench_tflite.py --model models/yolov8n_vehicle_320_int8.tflite --name int8-cpu
  # NPU through the VX delegate
  python3 bench_tflite.py --model models/yolov8n_vehicle_320_int8.tflite --name int8-npu \
          --delegate /usr/lib/libvx_delegate.so
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from collab import iou_matrix, match_counts
from tflite_utils import TfliteDetector


def ap50(dets_list, gts_list):
    recs, n_gt = [], sum(len(g) for g in gts_list)
    for d, g in zip(dets_list, gts_list):
        used = set()
        for i in np.argsort(-d[:, 4]) if len(d) else []:
            best, bj = 0.0, -1
            if len(g):
                ious = iou_matrix(d[i:i + 1, :4], g)[0]
                for j in range(len(g)):
                    if j not in used and ious[j] > best:
                        best, bj = ious[j], j
            tp = bj >= 0 and best >= 0.5
            if tp:
                used.add(bj)
            recs.append((d[i, 4], tp))
    if not recs or n_gt == 0:
        return 0.0
    recs.sort(key=lambda r: -r[0])
    tp = np.cumsum([r[1] for r in recs]); fp = np.cumsum([not r[1] for r in recs])
    rec, prec = tp / n_gt, tp / np.maximum(tp + fp, 1)
    return float(np.mean([prec[rec >= t].max() if (rec >= t).any() else 0 for t in np.linspace(0, 1, 101)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--delegate", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--pack", default="data/pack")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="0 = all images")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--coco-vehicles", action="store_true",
                    help="COCO-pretrained model: keep only bicycle/car/motorcycle/bus/truck and treat them as one class")
    args = ap.parse_args()

    imgs = np.load(Path(args.pack) / "pack.npz")["images"]
    meta = json.load(open(Path(args.pack) / "pack.json"))["images"]
    n = len(imgs) if not args.limit else min(args.limit, len(imgs))

    det = TfliteDetector(args.model, args.delegate, args.threads)
    t = time.perf_counter(); det.run(imgs[0]); cold = (time.perf_counter() - t) * 1000   # includes NPU graph compile
    for _ in range(args.warmup):
        det.run(imgs[0])

    t_pre, t_inv, t_post, d25, d05, gts = [], [], [], [], [], []
    for i in range(n):
        out, tp, ti = det.run(imgs[i])
        t0 = time.perf_counter(); d = det.decode(out, args.conf); t_post.append((time.perf_counter() - t0) * 1000)
        t_pre.append(tp); t_inv.append(ti)
        d05_i = det.decode(out, 0.05)
        if args.coco_vehicles:
            d = d[np.isin(d[:, 5], [1, 2, 3, 5, 7])]; d05_i = d05_i[np.isin(d05_i[:, 5], [1, 2, 3, 5, 7])]
        d[:, 5] = 0; d05_i[:, 5] = 0                    # single 'vehicle' class
        d25.append(d); d05.append(d05_i)
        gts.append(np.array(meta[i]["gt"], float).reshape(-1, 4))

    tot = np.array(t_pre) + np.array(t_inv) + np.array(t_post)
    tp_ = fp_ = fn_ = 0
    for d, g in zip(d25, gts):
        a, b, c = match_counts(d, np.concatenate([np.zeros((len(g), 1)), g], 1)); tp_ += a; fp_ += b; fn_ += c
    P, R = tp_ / max(tp_ + fp_, 1), tp_ / max(tp_ + fn_, 1)
    row = dict(name=args.name, model=Path(args.model).name, backend="NPU(VX)" if args.delegate else f"CPU x{args.threads}",
               size_mb=Path(args.model).stat().st_size / 1e6, cold_ms=cold, pre=np.mean(t_pre), invoke=np.mean(t_inv),
               post=np.mean(t_post), mean=tot.mean(), p95=np.percentile(tot, 95), fps=1000 / tot.mean(),
               P=P, R=R, F1=2 * P * R / max(P + R, 1e-9), AP50=ap50(d05, gts), n=n)
    print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}, indent=1))

    md = Path("results/benchmark.md"); md.parent.mkdir(exist_ok=True)
    if not md.exists():
        md.write_text("| name | backend | size MB | cold start ms | preproc ms | invoke ms | postproc ms | total ms | p95 ms | FPS | P | R | F1 | AP50 | imgs |\n"
                      "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    with open(md, "a") as f:
        f.write(f"| {row['name']} | {row['backend']} | {row['size_mb']:.2f} | {cold:.0f} | {row['pre']:.2f} | {row['invoke']:.2f} | "
                f"{row['post']:.2f} | {row['mean']:.2f} | {row['p95']:.2f} | {row['fps']:.1f} | {P:.3f} | {R:.3f} | "
                f"{row['F1']:.3f} | {row['AP50']:.3f} | {n} |\n")


if __name__ == "__main__":
    main()

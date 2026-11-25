"""Live demo on the board: same vehicle in two views?  (collaborative perception with cross-view association)

Mode 1 - nuScenes pairs (calibrated, with ground truth):
  python3 demo_assoc.py --pack pack --model ../ft_int8.tflite --delegate /usr/lib/libvx_delegate.so --limit 12
Mode 2 - your own two photos of the same scene (real-world controlled test, no calibration -> colour + keypoints):
  python3 demo_assoc.py --pair photoA.jpg photoB.jpg --model ../ft_int8.tflite --delegate /usr/lib/libvx_delegate.so

For every pair it prints how many vehicles each view sees, how many are the SAME vehicle, the true count of
distinct vehicles (mode 1), and the full per-stage latency. Images go to results/assoc/:
top = both views (same colour + line = same vehicle, grey = seen by one view only),
bottom-left = top-down (BEV) map, bottom-right = numbers.
"""
import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import assoc
from detect_utils import VehicleDetector
from eval_assoc import assign_gt

PALETTE = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240),
           (240, 50, 230), (210, 245, 60), (0, 128, 128), (170, 110, 40)]
GREY = (150, 150, 150)
NAMES = assoc.CLASS_NAMES


def font(size):
    try:
        return ImageFont.load_default(size=size)       # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def jpeg_size(img, q=80):
    b = io.BytesIO(); img.save(b, "JPEG", quality=q); return b.tell()


def draw_views(imgA, imgB, va, vb, res, gap=8):
    W, H = imgA.size
    canvas = Image.new("RGB", (2 * W + gap, H), (20, 20, 20))
    canvas.paste(imgA, (0, 0)); canvas.paste(imgB, (W + gap, 0))
    dr = ImageDraw.Draw(canvas)
    matched_a = {i: (k, j) for k, (i, j, _, _) in enumerate(res["matches"])}
    matched_b = {j: k for k, (i, j, _, _) in enumerate(res["matches"])}
    for i, d in enumerate(va.dets):
        col = PALETTE[matched_a[i][0] % len(PALETTE)] if i in matched_a else GREY
        dr.rectangle([d[0], d[1], d[2], d[3]], outline=col, width=3)
        tag = f"#{matched_a[i][0] + 1} {NAMES[int(d[5])]}" if i in matched_a else f"A {NAMES[int(d[5])]}"
        dr.text((d[0] + 3, max(0, d[1] - 18)), tag, fill=col, font=font(16))
    for j, d in enumerate(vb.dets):
        col = PALETTE[matched_b[j] % len(PALETTE)] if j in matched_b else GREY
        x1, x2 = d[0] + W + gap, d[2] + W + gap
        dr.rectangle([x1, d[1], x2, d[3]], outline=col, width=3)
        tag = f"#{matched_b[j] + 1} {NAMES[int(d[5])]}" if j in matched_b else f"B {NAMES[int(d[5])]}"
        dr.text((x1 + 3, max(0, d[1] - 18)), tag, fill=col, font=font(16))
    for k, (i, j, cost, _) in enumerate(res["matches"]):
        a, b = va.dets[i], vb.dets[j]
        dr.line([((a[0] + a[2]) / 2, (a[1] + a[3]) / 2), ((b[0] + b[2]) / 2 + W + gap, (b[1] + b[3]) / 2)],
                fill=PALETTE[k % len(PALETTE)], width=2)
    return canvas


def draw_bev(va, vb, res, objs, size=380, span=45.0):
    img = Image.new("RGB", (size, size), (25, 25, 30))
    dr = ImageDraw.Draw(img)
    s = size / (2 * span)

    def px(x, y):                       # ego frame (x forward, y left) -> pixels (up = forward)
        return size / 2 - y * s, size / 2 - x * s

    for r in (10, 20, 30, 40):
        dr.ellipse([size / 2 - r * s, size / 2 - r * s, size / 2 + r * s, size / 2 + r * s], outline=(55, 55, 65))
    for v, col in ((va, (255, 140, 0)), (vb, (0, 140, 255))):             # field of view of each camera
        o, d = assoc.rays(v, [0, v.W], [v.H / 2, v.H / 2])
        for k in range(2):
            ang = np.arctan2(d[k, 1], d[k, 0])
            dr.line([px(*o[:2]), px(o[0] + span * 1.4 * np.cos(ang), o[1] + span * 1.4 * np.sin(ang))], fill=col, width=1)
    dr.rectangle([px(-1.0, 1.0)[0], px(2.0, 1.0)[1], px(-1.0, -1.0)[0], px(-1.0, -1.0)[1]], fill=(230, 230, 230))
    k = 0
    for o in objs:
        if o["foot"] is None:
            continue
        x, y = px(*o["foot"])
        if o["views"] == "AB":
            col = PALETTE[k % len(PALETTE)]; k += 1
            dr.ellipse([x - 7, y - 7, x + 7, y + 7], fill=col, outline=(255, 255, 255), width=2)
        else:
            col = (255, 140, 0) if o["views"] == "A" else (0, 140, 255)
            dr.ellipse([x - 6, y - 6, x + 6, y + 6], outline=col, width=2)
    dr.text((6, 4), "BEV (ego car = white, 45 m)", fill=(200, 200, 200), font=font(14))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="pack")
    ap.add_argument("--pair", nargs=2, metavar=("A", "B"), help="two photos of the same scene (no calibration)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--delegate")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--cues", default="geo,app,kp")
    ap.add_argument("--params", default="assoc_params.json", help="tuned weights from eval_assoc.py --tune (optional)")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--only-shared", action="store_true", help="skip pairs without a vehicle seen by both views")
    ap.add_argument("--pose-noise-m", type=float, default=0.0)
    ap.add_argument("--pose-noise-deg", type=float, default=0.0)
    ap.add_argument("--out", default="results/assoc")
    args = ap.parse_args()

    det = VehicleDetector(args.model, args.delegate)
    print(f"Detector: {args.model} | {'NPU (VX delegate)' if args.delegate else 'CPU'} | {det.det.imgsz}x{det.det.imgsz} | {det.nc} classes")
    cues = tuple(args.cues.split(","))
    weights, thr = None, assoc.DEFAULT_THR
    if Path(args.params).exists() and not args.pair:
        pr = json.loads(Path(args.params).read_text())
        cues, weights, thr = tuple(pr["cues"]), pr["weights"], pr["thr"]
        print(f"Using tuned association parameters from {args.params}: {pr}")
    if "kp" in cues:
        try:
            import cv2  # noqa: F401
        except ImportError:
            print("NOTE: OpenCV is not installed on this machine -> the keypoint cue is skipped (geometry + colour only).")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    jobs = []
    if args.pair:
        jobs.append({"name": "photos", "imgA": args.pair[0], "imgB": args.pair[1], "pair": None})
    else:
        data = json.loads((Path(args.pack) / "pairs.json").read_text())
        for n, p in enumerate(data["pairs"]):
            if args.only_shared and p["n_shared"] == 0:
                continue
            jobs.append({"name": f"{n:03d}_{p['sample'][:6]}_{p['camA'][4:]}-{p['camB'][4:]}",
                         "imgA": Path(args.pack) / p["imgA"], "imgB": Path(args.pack) / p["imgB"], "pair": p})
        jobs = jobs[: args.limit] if args.limit else jobs

    T = {"letterbox": [], "quantize": [], "invoke": [], "decode": [], "assoc": []}
    tot_naive = tot_assoc = tot_tp = tot_wrong = tot_unver = tot_gtpairs = 0
    jpg_bytes = msg_bytes = 0
    for job in jobs:
        imA = Image.open(job["imgA"]).convert("RGB"); imB = Image.open(job["imgB"]).convert("RGB")
        dA, tA = det(imA, args.conf); dB, tB = det(imB, args.conf)
        p = job["pair"]
        if p:
            size = tuple(p["size"]); calibrated = True
            va = assoc.View(p["KA"], p["TA"], dA, size, np.asarray(imA), p["camA"])
            vb = assoc.View(p["KB"], p["TB"], dB, size, np.asarray(imB), p["camB"])
            vb_est = assoc.perturb_pose(vb, rng, args.pose_noise_m, args.pose_noise_deg)
            use = cues
        else:
            K = np.array([[imA.size[0], 0, imA.size[0] / 2], [0, imA.size[0], imA.size[1] / 2], [0, 0, 1.0]])
            va = assoc.View(K, np.eye(4), dA, imA.size, np.asarray(imA), "A")
            vb = assoc.View(K, np.eye(4), dB, imB.size, np.asarray(imB), "B")
            vb_est, calibrated, use = vb, False, tuple(c for c in cues if c != "geo")
        t = time.perf_counter()
        res = assoc.associate(va, vb_est, cues=use, weights=weights, thr=thr, calibrated=calibrated)
        t_assoc = (time.perf_counter() - t) * 1e3
        objs = assoc.merge_objects(va, vb_est, res)
        for k in ("letterbox", "quantize", "invoke", "decode"):
            T[k] += [tA[k], tB[k]]
        T["assoc"].append(t_assoc)

        nA, nB, nM = len(dA), len(dB), len(res["matches"])
        line = f"{job['name']:<34} A={nA:<2} B={nB:<2} same-vehicle pairs={nM} | count: naive {nA + nB} -> merged {nA + nB - nM}"
        if p:
            ia, ib = assign_gt(dA, p["objsA"]), assign_gt(dB, p["objsB"])
            true = {(i, j) for i, a in enumerate(ia) for j, b in enumerate(ib) if a >= 0 and a == b}
            pred = {(i, j) for i, j, _, _ in res["matches"]}
            both = {(i, j) for (i, j) in pred if ia[i] >= 0 and ib[j] >= 0}
            wrong = len(both - true)
            line += (f" | correct links {len(pred & true)}/{len(true)}" + (f", WRONG {wrong}" if wrong else "")
                     + (f", unverifiable {len(pred) - len(both)}" if len(pred) - len(both) else ""))
            tot_tp += len(pred & true); tot_wrong += wrong; tot_unver += len(pred) - len(both); tot_gtpairs += len(true)
        tot_naive += nA + nB; tot_assoc += nA + nB - nM
        print(line)
        jpg_bytes += jpeg_size(imA) + jpeg_size(imB); msg_bytes += assoc.message_bytes(dA) + assoc.message_bytes(dB)
        canvas = draw_views(imA, imB, va, vb, res)
        W, H = canvas.size
        full = Image.new("RGB", (W, H + 380), (20, 20, 20)); full.paste(canvas, (0, 0))
        if calibrated:
            full.paste(draw_bev(va, vb_est, res, objs), (0, H))
        dr = ImageDraw.Draw(full)
        txt = [f"{job['name']}", f"A sees {nA}, B sees {nB}", f"same vehicle in both views: {nM}",
               f"vehicles counted without association: {nA + nB}", f"vehicles after association: {nA + nB - nM}",
               f"association cost: {t_assoc:.1f} ms (cues: {'+'.join(use)})"]
        if p:
            txt.insert(5, f"ground-truth links found: {len(pred & true)} of {len(true)}")
        for n, s in enumerate(txt):
            dr.text((400, H + 14 + 28 * n), s, fill=(235, 235, 235), font=font(20))
        full.save(out / f"{job['name']}.jpg", quality=88)

    n = max(len(jobs), 1)
    per_img = sum(np.mean(T[k]) for k in ("letterbox", "quantize", "invoke", "decode"))
    pipe = 2 * per_img + np.mean(T["assoc"])
    print("\n=== SUMMARY ===")
    print(f"pairs: {len(jobs)} | boxes counted without association: {tot_naive} -> after association: {tot_assoc}"
)
    if tot_gtpairs:
        pr, rc = tot_tp / max(tot_tp + tot_wrong, 1), tot_tp / tot_gtpairs
        print(f"same-vehicle links: verified precision {pr:.2f}, recall {rc:.2f} ({tot_tp} correct of {tot_gtpairs} true links, "
              f"{tot_wrong} wrong, {tot_unver} to boxes without a ground-truth match = not judged)")
    print("latency per image (mean ms): letterbox %.1f | quantise %.1f | invoke %.1f | dequant+decode+NMS %.1f = %.1f" %
          (np.mean(T["letterbox"]), np.mean(T["quantize"]), np.mean(T["invoke"]), np.mean(T["decode"]), per_img))
    print(f"association per pair: {np.mean(T['assoc']):.1f} ms | full pair pipeline (2 detections + association): "
          f"{pipe:.1f} ms = {1000 / pipe:.1f} pairs/s")
    print(f"uplink per pair: JPEG {jpg_bytes / n / 1e3:.1f} kB vs detection messages {msg_bytes / n:.0f} B")
    print(f"images saved in {out}/")


if __name__ == "__main__":
    main()

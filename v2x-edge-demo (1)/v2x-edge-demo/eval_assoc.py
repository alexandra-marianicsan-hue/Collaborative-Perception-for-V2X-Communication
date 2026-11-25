"""Evaluate cross-view association on the nuScenes pack made by prepare_nuscenes.py.

Question answered: when the same vehicle appears in two neighbouring cameras (often cut in half), does the
system realise it is ONE vehicle?  Ground truth = nuScenes instance_token.

Detections: --dets oracle  (ground-truth boxes + localisation noise: isolates the association step; runs anywhere)
            --dets model   (real TFLite detector: --model ... [--delegate /usr/lib/libvx_delegate.so]; on the board)
  python3 eval_assoc.py --pack data/nuscenes_v2x/pack --dets oracle
  python3 eval_assoc.py --pack pack --dets model --model ft_int8.tflite --delegate /usr/lib/libvx_delegate.so --tune
Metrics (pairs of boxes (i in A, j in B) that are the same vehicle):
  precision / recall / F1, recall on CUT objects (box touches the image border), and the object-count error:
  'naive' counts every box (a cut bus is counted twice), 'assoc' merges the associated boxes.
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

import assoc
from collab import iou_matrix

CONFIGS = [
    ("no association", None),
    ("geo", ("geo",)),
    ("app", ("app",)),
    ("kp", ("kp",)),
    ("geo+app", ("geo", "app")),
    ("geo+kp", ("geo", "kp")),
    ("geo+app+kp", ("geo", "app", "kp")),
]


def assign_gt(dets, objs, thr=0.4):
    """Greedy IoU assignment of detections to GT boxes -> instance id per detection (-1 = false positive)."""
    ids = [-1] * len(dets)
    if len(dets) == 0 or not objs:
        return ids
    iou = iou_matrix(dets[:, :4], np.array([o["box"] for o in objs]))
    cand = sorted(((iou[i, j], i, j) for i in range(len(dets)) for j in range(len(objs)) if iou[i, j] >= thr), reverse=True)
    ud, ug = set(), set()
    for _, i, j in cand:
        if i not in ud and j not in ug:
            ids[i] = objs[j]["inst"]; ud.add(i); ug.add(j)
    return ids


def oracle_dets(objs, size, rng, jitter=0.03):
    out = []
    for o in objs:
        b = np.array(o["box"], dtype=np.float64)
        wh = np.array([b[2] - b[0], b[3] - b[1]] * 2)
        b = np.clip(b + rng.normal(0, jitter, 4) * wh, 0, [size[0], size[1]] * 2)
        out.append([*b, 0.9, o["cls"]])
    return np.array(out, dtype=np.float64).reshape(-1, 6)


def load_pairs(pack, dets_mode, detector, conf, seed=0):
    data = json.loads((Path(pack) / "pairs.json").read_text())
    rng = np.random.default_rng(seed)
    items = []
    for p in data["pairs"]:
        size = tuple(p["size"])
        imgs, dets, ids, tm = [], [], [], []
        for side in "AB":
            img = Image.open(Path(pack) / p["img" + side]).convert("RGB")
            objs = p["objs" + side]
            if dets_mode == "oracle":
                d, t = oracle_dets(objs, size, rng), {}
            else:
                d, t = detector(img, conf)
            imgs.append(np.asarray(img)); dets.append(d); ids.append(assign_gt(d, objs)); tm.append(t)
        va = assoc.View(p["KA"], p["TA"], dets[0], size, imgs[0], p["camA"])
        vb = assoc.View(p["KB"], p["TB"], dets[1], size, imgs[1], p["camB"])
        items.append({"va": va, "vb": vb, "ia": ids[0], "ib": ids[1], "timing": tm, "pair": p})
    return items


def score(items, cues, weights=None, thr=assoc.DEFAULT_THR, pose_noise=(0, 0), seed=1):
    """Link metrics. A predicted link is
         correct      both boxes matched to the same GT vehicle,
         wrong        both boxes matched to GT but to DIFFERENT vehicles,
         unverifiable at least one box has no GT match (the detector found a vehicle nobody annotated:
                      far, parked, occluded...). These are NOT counted as errors in the verified precision.
    Count error is computed on the GT-matched boxes only (true number of distinct vehicles = distinct GT ids)."""
    rng = np.random.default_rng(seed)
    tp = wrong = unver = fn = 0
    cut_tp = cut_fn = 0
    naive_err = assoc_err = 0
    ms = []
    for it in items:
        va, vb, ia, ib = it["va"], it["vb"], it["ia"], it["ib"]
        true = {(i, j) for i, a in enumerate(ia) for j, b in enumerate(ib) if a >= 0 and a == b}
        if cues is None:
            pred = set()
        else:
            vb_est = assoc.perturb_pose(vb, rng, *pose_noise)
            t = time.perf_counter()
            res = assoc.associate(va, vb_est, cues=cues, weights=weights, thr=thr)
            ms.append((time.perf_counter() - t) * 1e3)
            pred = {(i, j) for i, j, _, _ in res["matches"]}
        both = {(i, j) for (i, j) in pred if ia[i] >= 0 and ib[j] >= 0}
        tp += len(pred & true); wrong += len(both - true); unver += len(pred) - len(both); fn += len(true - pred)
        for (i, j) in true:
            da, db = va.dets[i], vb.dets[j]
            cut = da[0] <= 4 or da[2] >= va.W - 4 or db[0] <= 4 or db[2] >= vb.W - 4
            if cut:
                cut_tp += (i, j) in pred; cut_fn += (i, j) not in pred
        m = sum(1 for v in ia + ib if v >= 0)                     # boxes matched to GT
        true_unique = len({v for v in ia + ib if v >= 0})
        naive_err += abs(m - true_unique)
        assoc_err += abs(m - len(both) - true_unique)
    p = tp / max(tp + wrong, 1); r = tp / max(tp + fn, 1)
    return {"P": p, "R": r, "F1": 2 * p * r / max(p + r, 1e-9), "cutR": cut_tp / max(cut_tp + cut_fn, 1),
            "cut_n": cut_tp + cut_fn, "n_true": tp + fn, "wrong": wrong, "unver": unver,
            "err": assoc_err, "naive_err": naive_err, "ms": float(np.mean(ms)) if ms else 0.0}


def tune(items, cues):
    """Small grid search of cue weights and acceptance threshold (maximise F1)."""
    best = (-1, None)
    grid_app = [0.0, 0.25, 0.5, 1.0] if "app" in cues else [0.0]
    grid_kp = [0.0, 0.5, 1.0, 2.0] if "kp" in cues else [0.0]
    for wa, wk, thr in itertools.product(grid_app, grid_kp, [0.35, 0.45, 0.55, 0.65, 0.75]):
        w = {"geo": 1.0, "app": wa or 1e-9, "kp": wk or 1e-9}
        c = tuple(x for x in cues if x == "geo" or (x == "app" and wa) or (x == "kp" and wk))
        s = score(items, c, w, thr)
        if s["F1"] >= best[0]:                           # ties -> later = more cues / looser threshold
            best = (s["F1"], {"cues": list(c), "weights": {"geo": 1.0, "app": wa, "kp": wk}, "thr": thr})
    return best[1]


def table(rows, title):
    lines = [f"\n{title}",
             "| method | P (verified) | R | F1 | wrong links | links to unannotated boxes | recall on cut objects | count error | ms/pair |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, s in rows:
        lines.append(f"| {name} | {s['P']:.2f} | {s['R']:.2f} | {s['F1']:.2f} | {s['wrong']} | {s['unver']} | "
                     f"{s['cutR']:.2f} (n={s['cut_n']}) | {s['naive_err'] if name == 'no association' else s['err']} | {s['ms']:.1f} |")
    lines.append("(P verified = correct / (correct + wrong), links to boxes without a ground-truth match are not judged; "
                 "count error on GT-matched boxes only)")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="data/nuscenes_v2x/pack")
    ap.add_argument("--dets", default="oracle", choices=["oracle", "model"])
    ap.add_argument("--model")
    ap.add_argument("--delegate")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--pose-noise-m", type=float, default=0.0)
    ap.add_argument("--pose-noise-deg", type=float, default=0.0)
    ap.add_argument("--tune", action="store_true", help="tune weights on the first half, report on the second half")
    ap.add_argument("--out", default="results/assoc_eval.md")
    args = ap.parse_args()

    detector = None
    if args.dets == "model":
        from detect_utils import VehicleDetector
        detector = VehicleDetector(args.model, args.delegate)
    items = load_pairs(args.pack, args.dets, detector, args.conf)
    n_pos = sum(1 for it in items if it["pair"]["n_shared"] > 0)
    print(f"{len(items)} camera pairs ({n_pos} with shared vehicles) | detections: {args.dets}"
          + (f" ({args.model})" if args.model else ""))
    if detector:
        tm = [t for it in items for t in it["timing"]]
        print("detector per image (mean ms): " + ", ".join(f"{k} {np.mean([t[k] for t in tm]):.1f}" for k in tm[0]))
        n_det = sum(len(it["va"].dets) + len(it["vb"].dets) for it in items)
        n_gt = sum(1 for it in items for v in it["ia"] + it["ib"] if v >= 0)
        print(f"detections: {n_det} total, {n_gt} matched to a ground-truth vehicle ({100 * n_gt / max(n_det, 1):.0f}%)")

    pn = (args.pose_noise_m, args.pose_noise_deg)
    out_txt = []
    eval_items, params = items, {}
    if args.tune:
        half = len(items) // 2
        tune_items, eval_items = items[:half], items[half:]
        params = tune(tune_items, ("geo", "app", "kp"))
        print("tuned on first half:", params)
        Path("assoc_params.json").write_text(json.dumps(params))
    rows = []
    for name, cues in CONFIGS:
        if name == "geo+app+kp" and params:
            s = score(eval_items, tuple(params["cues"]), params["weights"], params["thr"], pn)
            name = "geo+app+kp (tuned)"
        else:
            s = score(eval_items, cues, None, assoc.DEFAULT_THR, pn)
        rows.append((name if name != "no association" else name, s))
    out_txt.append(table(rows, f"Association on {len(eval_items)} pairs, pose noise {pn[0]} m / {pn[1]} deg, "
                               f"detections={args.dets}"))
    print("\n".join(out_txt))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a") as f:
        f.write("\n".join(out_txt) + "\n")
    print(f"\n(appended to {args.out})")


if __name__ == "__main__":
    main()

"""Collaborative perception on OPV2V, running on the edge board (the LIVE DEMO).

For every timestamp, each CAV runs the TFLite detector (NPU) on its front camera. Detections
are back-projected to world (BEV) coordinates and only a tiny message (x, y, conf) is 'sent'.
The ego vehicle fuses its own detections with the received ones. We compare:
   ego alone   vs   ego + collaborators   (recall of ground-truth vehicles within --range m)

  python3 collab_opv2v.py --model models/yolov8n_vehicle_320_int8.tflite \
          --delegate /usr/lib/libvx_delegate.so --sleep 0.5
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from opv2v_geometry import backproject_ground
from tflite_utils import TfliteDetector


def det_to_world(d, m, ground_offset, center_offset=2.0, max_range=60.0):
    """Bottom-centre of the 2D box -> point on the ground -> shifted ~half a car length away from the camera."""
    x1, y1, x2, y2, conf, _ = d
    u, v = ((x1 + x2) / 2 - m["left"]) / m["r"], (y2 - m["top"]) / m["r"]
    pose = m["pose"]
    p = backproject_ground(u, v, pose, m["K"], pose[2] + ground_offset)
    if p is None:
        return None
    vec = p[:2] - np.array(pose[:2]); dist = np.linalg.norm(vec)
    if dist < 1.0 or dist > max_range:
        return None
    q = p[:2] + vec / dist * center_offset
    return [q[0], q[1], conf]


def fuse_world(dets, merge=2.5):
    """dets: (N,3) x,y,conf from all CAVs -> drop duplicates closer than `merge` metres (keep best conf)."""
    dets = np.asarray(dets, float).reshape(-1, 3)
    keep = []
    for i in np.argsort(-dets[:, 2]) if len(dets) else []:
        if all(np.hypot(*(dets[i, :2] - dets[j, :2])) >= merge for j in keep):
            keep.append(i)
    return dets[keep]


def match_bev(dets, gt, thr=3.0):
    """Greedy nearest matching. Returns (matched GT mask, tp)."""
    used = np.zeros(len(gt), bool); tp = 0
    for i in np.argsort(-dets[:, 2]) if len(dets) else []:
        if not len(gt):
            break
        dist = np.hypot(*(gt - dets[i, :2]).T); dist[used] = 1e9
        j = dist.argmin()
        if dist[j] <= thr:
            used[j] = True; tp += 1
    return used, tp


def draw_bev(path, ego, gt, ego_d, fused, R, scale=5):
    import cv2
    size = int(2 * R * scale)
    img = np.full((size, size, 3), 30, np.uint8)
    px = lambda p: (int((p[0] - ego[0]) * scale + size / 2), int((p[1] - ego[1]) * scale + size / 2))
    for g in gt: cv2.circle(img, px(g), 6, (160, 160, 160), 1)
    for d in fused: cv2.circle(img, px(d), 4, (0, 140, 255), -1)
    for d in ego_d: cv2.circle(img, px(d), 3, (255, 140, 0), -1)
    cv2.circle(img, px(ego), 5, (0, 255, 0), -1)
    cv2.putText(img, "grey=GT  blue=ego only  orange=fused (collab)", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.imwrite(str(path), img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--delegate", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--pack", default="data/pack")
    ap.add_argument("--range", type=float, default=50.0)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between timestamps (live feel)")
    ap.add_argument("--save-vis", default=None, help="folder for BEV pngs (needs OpenCV)")
    args = ap.parse_args()

    imgs = np.load(Path(args.pack) / "pack.npz")["images"]
    pk = json.load(open(Path(args.pack) / "pack.json"))
    det = TfliteDetector(args.model, args.delegate, args.threads)
    det.run(imgs[0])                                                    # warm-up (NPU graph compile)
    if args.save_vis:
        Path(args.save_vis).mkdir(parents=True, exist_ok=True)
    print(f"model={args.model} backend={'NPU' if args.delegate else 'CPU'} timestamps={len(pk['timestamps'])}")

    tot = dict(gt=0, ego=0, fused=0, det_ego=0, det_fused=0, tp_ego=0, tp_fused=0)
    lat, bytes_msg, bytes_raw = [], 0, 0
    for k, t in enumerate(pk["timestamps"]):
        per_cav, ego_d = {}, np.zeros((0, 3))
        for i in t["image_ids"]:
            m = pk["images"][i]
            out, a, b = det.run(imgs[i]); lat.append(a + b)
            ds = [det_to_world(d, m, pk["ground_offset"]) for d in det.decode(out, args.conf)]
            per_cav[m["cav"]] = np.array([d for d in ds if d is not None]).reshape(-1, 3)
            bytes_msg += 4 + 9 * len(per_cav[m["cav"]]); bytes_raw += m["W"] * m["H"] * 3
        ego_d = per_cav[t["ego"]]
        fused = fuse_world(np.concatenate(list(per_cav.values()), 0))
        ego_m = pk["images"][[i for i in t["image_ids"] if pk["images"][i]["ego"]][0]]
        ego_xy = np.array(ego_m["pose"][:2])
        gt = np.array([[g["x"], g["y"]] for g in t["gt_world"]]).reshape(-1, 2)
        gt = gt[np.hypot(*(gt - ego_xy).T) <= args.range] if len(gt) else gt
        inr = lambda d: d[np.hypot(*(d[:, :2] - ego_xy).T) <= args.range] if len(d) else d
        e, f = inr(ego_d), inr(fused)
        me, te = match_bev(e, gt); mf, tf = match_bev(f, gt)
        for key, val in dict(gt=len(gt), ego=me.sum(), fused=mf.sum(), det_ego=len(e), det_fused=len(f), tp_ego=te, tp_fused=tf).items():
            tot[key] += int(val)
        print(f"[{k:03d}] {t['scenario']}/{t['ts']} CAVs={len(per_cav)} GT={len(gt)} "
              f"found: ego-only={me.sum()} collaborative={mf.sum()}")
        if args.save_vis:
            draw_bev(Path(args.save_vis) / f"bev_{k:04d}.png", ego_xy, gt, e[:, :2], f[:, :2], args.range)
        if args.sleep:
            time.sleep(args.sleep)

    rec_e, rec_f = tot["ego"] / max(tot["gt"], 1), tot["fused"] / max(tot["gt"], 1)
    prec_e, prec_f = tot["tp_ego"] / max(tot["det_ego"], 1), tot["tp_fused"] / max(tot["det_fused"], 1)
    txt = (f"# Collaborative perception on OPV2V (BEV, range {args.range:.0f} m, match 3 m)\n\n"
           f"| setup | recall | precision* |\n|---|---|---|\n| ego alone | {rec_e:.3f} | {prec_e:.3f} |\n"
           f"| ego + collaborators | {rec_f:.3f} | {prec_f:.3f} |\n\n"
           f"*GT only contains vehicles hit by LiDAR, so precision is indicative.\n\n"
           f"- inference per CAV frame: {np.mean(lat):.1f} ms ({'NPU' if args.delegate else 'CPU'}), p95 {np.percentile(lat, 95):.1f} ms\n"
           f"- uplink per frame: raw image {bytes_raw / len(lat) / 1e6:.2f} MB vs detections {bytes_msg / len(lat):.0f} B\n")
    print("\n" + txt)
    Path("results").mkdir(exist_ok=True)
    Path("results/collab_opv2v.md").write_text(txt)


if __name__ == "__main__":
    main()

"""Synthetic sanity test for assoc.py (no dataset needed).

Random vehicles around the ego car are projected into the six nuScenes-like cameras. For each pair of
adjacent cameras we feed the (jittered) 2D boxes to the associator and compare with the true identities.
  python3 test_assoc_synth.py            # geometry-only association
  python3 test_assoc_synth.py --kp       # also test the ORB keypoint cue on warped images
"""
import argparse

import numpy as np

import assoc
from nusc_geometry import PAIRS, RIG, box_corners, cam_to_ego, intrinsics, project_box

SIZES = {1: (1.9, 4.6, 1.7), 3: (2.9, 11.0, 3.3), 4: (2.6, 7.0, 2.9), 2: (0.8, 2.2, 1.5), 0: (0.6, 1.8, 1.5)}
W, H = 1600, 900


def make_world(rng, n=14):
    """Random vehicles that do not overlap each other (centre distance > half lengths + 1 m)."""
    objs = []
    tries = 0
    while len(objs) < n and tries < 2000:
        tries += 1
        r = rng.uniform(7, 45); a = rng.uniform(-np.pi, np.pi)
        cls = int(rng.choice([1, 1, 1, 1, 3, 4, 2, 0]))
        wlh = SIZES[cls]
        c = (r * np.cos(a), r * np.sin(a), wlh[2] / 2)
        if any(np.hypot(c[0] - o["c"][0], c[1] - o["c"][1]) < (wlh[1] + o["wlh"][1]) / 2 + 1.0 for o in objs):
            continue
        objs.append({"id": len(objs), "cls": cls, "c": c, "wlh": wlh, "yaw": rng.uniform(-np.pi, np.pi)})
    return objs


def detect_view(objs, cam, rng, jitter=0.03, min_vis=0.25):
    x, y, z, yaw, fx = RIG[cam]
    T, K = cam_to_ego(x, y, z, yaw), intrinsics(fx, W, H)
    inv = np.linalg.inv(T)
    dets, ids = [], []
    for o in objs:
        pc = inv[:3, :3] @ box_corners(o["c"], o["wlh"], o["yaw"]) + inv[:3, 3:4]
        pr = project_box(pc, K, W, H)
        if pr is None or pr["clip_area"] / max(pr["full_area"], 1) < min_vis:
            continue
        b = np.array(pr["box"])
        if b[2] - b[0] < 14 or b[3] - b[1] < 14:
            continue
        wh = np.array([b[2] - b[0], b[3] - b[1]] * 2)
        b = b + rng.normal(0, jitter, 4) * wh                        # detector localisation noise
        b = np.clip(b, 0, [W, H, W, H])
        dets.append([*b, 0.9, o["cls"]]); ids.append(o["id"])
    return assoc.View(K, T, np.array(dets).reshape(-1, 6), (W, H), name=cam), ids


def run(n_trials, sigma_m, sigma_deg, seed=0):
    rng = np.random.default_rng(seed)
    tp = fp = fn = 0
    trunc_tp = trunc_fn = 0
    naive_err = assoc_err = 0
    for _ in range(n_trials):
        objs = make_world(rng)
        for ca, cb in PAIRS:
            va, ia = detect_view(objs, ca, rng)
            vb, ib = detect_view(objs, cb, rng)
            true = {(i, j) for i, a in enumerate(ia) for j, b in enumerate(ib) if a == b}
            vb_est = assoc.perturb_pose(vb, rng, sigma_m, sigma_deg)
            res = assoc.associate(va, vb_est, cues=("geo",))
            pred = {(i, j) for i, j, _, _ in res["matches"]}
            tp += len(pred & true); fp += len(pred - true); fn += len(true - pred)
            for (i, j) in true:                                        # how well do we do on cut-in-half objects?
                if assoc.describe(va, va.dets[i])["trunc"] or assoc.describe(vb, vb.dets[j])["trunc"]:
                    trunc_tp += (i, j) in pred; trunc_fn += (i, j) not in pred
            uniq = len(set(ia) | set(ib))
            naive_err += abs(len(ia) + len(ib) - uniq)
            assoc_err += abs(len(ia) + len(ib) - len(pred) - uniq)
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1); f1 = 2 * p * r / max(p + r, 1e-9)
    tr = trunc_tp / max(trunc_tp + trunc_fn, 1)
    return p, r, f1, tr, naive_err, assoc_err, tp + fn


def kp_test():
    """ORB cue: warp one textured image into two rotated camera views; same-object boxes must collect matches."""
    import cv2
    rng = np.random.default_rng(1)
    big = (rng.random((900, 1600, 3)) * 255).astype(np.uint8)
    big = cv2.GaussianBlur(big, (0, 0), 1.2)
    K = intrinsics(1266.4, W, H)
    TA = cam_to_ego(1.7, 0, 1.5, 0)
    TB = cam_to_ego(1.5, 0.5, 1.5, 55)
    va = assoc.View(K, TA, [[60, 300, 360, 700, .9, 1], [1100, 300, 1500, 700, .9, 1]], (W, H), img=big)   # box 0 lies in the shared wedge (left edge of A)
    Hm = assoc.warp_b_to_a(va, assoc.View(K, TB, np.zeros((0, 6)), (W, H)))
    b_img = cv2.warpPerspective(big, np.linalg.inv(Hm), (W, H))          # what camera B would see of the same wall
    # same object seen by B: box mapped through the inverse homography
    def to_b(box):
        pts = np.float32([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]])
        q = cv2.perspectiveTransform(pts[None], np.linalg.inv(Hm))[0]
        return [q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()]
    vb = assoc.View(K, TB, [[*to_b([60, 300, 360, 700]), .9, 1], [900, 200, 1300, 600, .9, 1]], (W, H), img=b_img)
    M = assoc.keypoint_matrix(va, vb, calibrated=True)
    print("B box 0 =", np.round(vb.dets[0, :4]))
    print("keypoint matrix (rows = boxes of A, cols = boxes of B):\n", M)
    ok = M[0, 0] >= 4 and M[0, 1] == 0 and M[1].sum() == 0
    print("kp cue OK" if ok else "kp cue FAILED")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=150)
    ap.add_argument("--kp", action="store_true")
    a = ap.parse_args()
    print(f"{'pose noise':>16} | {'P':>5} {'R':>5} {'F1':>5} | recall on cut objects | count error (naive -> assoc)")
    for sm, sd in [(0, 0), (0.3, 0.5), (0.5, 1.0), (1.0, 2.0)]:
        p, r, f1, tr, ne, ae, npairs = run(a.trials, sm, sd)
        print(f"{sm:>6.1f} m {sd:>4.1f} deg | {p:5.2f} {r:5.2f} {f1:5.2f} | {tr:21.2f} | {ne:>6} -> {ae:<6} ({npairs} true pairs)")
    if a.kp:
        kp_test()

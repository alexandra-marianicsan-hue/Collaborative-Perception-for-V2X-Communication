"""Cross-view association: "is the box in view A the same vehicle as the box in view B?"

Two vehicles (or two cameras) see an overlapping part of the scene. A bus can be cut in half:
half of it in image A, the other half in image B. To merge them into ONE object we combine
three independent cues, each cheap enough for the edge board:

  geo : geometric consistency. Each detection is lifted into a common ground-plane frame
        (BEV) with the shared camera pose + calibration. Same object -> overlapping bearing
        interval and nearby ground footprint (tolerance grows for truncated boxes).
  app : appearance. HSV colour histogram of the box (Bhattacharyya distance).
  kp  : visual evidence. ORB keypoint matches between the two images (view B warped into view A
        when the rotation is known). Matches that fall inside box i of A and box j of B vote for
        the pair (i, j).  Needs OpenCV (optional; skipped when cv2 is not installed).

Costs are combined, a class gate removes impossible pairs, and a greedy one-to-one matching
(no scipy needed) gives the associations. Merged pairs count as one object.

Coordinates: ego frame x forward, y left, z up. A View stores K (3x3) and T (4x4, camera->ego),
the camera frame being x right, y down, z forward (nuScenes / OpenCV convention).
Detections are (N, 6): x1, y1, x2, y2, conf, cls in pixels.
"""
import time

import numpy as np
from PIL import Image

CLASS_NAMES = {0: "bicycle", 1: "car", 2: "motorcycle", 3: "bus", 4: "truck"}
CLS_LEN = {0: 1.8, 1: 4.6, 2: 2.2, 3: 11.0, 4: 7.0}          # typical length [m] -> truncation tolerance
BIG = {1, 3, 4}                                                 # car / bus / truck can be confused with each other

DEFAULT_W = {"geo": 1.0, "app": 0.5, "kp": 1.0}
DEFAULT_THR = 0.55


class View:
    """One camera: image (optional), intrinsics, pose, detections."""

    def __init__(self, K, T, dets, size, img=None, name=""):
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.T = np.asarray(T, dtype=np.float64).reshape(4, 4)
        self.dets = np.asarray(dets, dtype=np.float64).reshape(-1, 6)
        self.W, self.H = int(size[0]), int(size[1])
        self.img = img                      # HxWx3 uint8 RGB or None
        self.name = name


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rays(view, u, v):
    u = np.atleast_1d(np.asarray(u, dtype=np.float64)); v = np.atleast_1d(np.asarray(v, dtype=np.float64))
    pix = np.stack([u, v, np.ones_like(u)], 1)
    d = (pix @ np.linalg.inv(view.K).T) @ view.T[:3, :3].T
    return view.T[:3, 3], d


def ground_point(view, u, v, z0=0.0):
    """Intersection of the pixel ray with the plane z = z0 (ego frame). NaN if the ray does not hit it."""
    o, d = rays(view, u, v)
    out = np.full((len(d), 2), np.nan)
    ok = d[:, 2] < -1e-3
    s = (z0 - o[2]) / d[ok, 2]
    out[ok] = o[:2] + s[:, None] * d[ok, :2]
    return out


def describe(view, det, border=4, max_range=80.0):
    """Geometry of one detection in the common frame."""
    x1, y1, x2, y2 = det[:4]
    trunc_l, trunc_r, trunc_b = x1 <= border, x2 >= view.W - border, y2 >= view.H - border
    foot, depth = None, np.nan
    if not trunc_b:
        g = ground_point(view, (x1 + x2) / 2, y2)[0]
        if np.isfinite(g).all():
            dep = float(np.hypot(*(g - view.T[:2, 3])))
            if dep <= max_range:
                foot, depth = g, dep
    o, d = rays(view, [x1, x2, (x1 + x2) / 2], [(y1 + y2) / 2] * 3)
    az = np.arctan2(d[:, 1], d[:, 0])
    c = az[2]
    a1, a2 = c + wrap(az[0] - c), c + wrap(az[1] - c)
    return {"foot": foot, "depth": depth, "az": (min(a1, a2), max(a1, a2)),
            "trunc": bool(trunc_l or trunc_r), "trunc_b": bool(trunc_b)}


def interval_overlap(a, b):
    """Overlap length of two angular intervals (b is shifted by 2*pi*k to be closest to a)."""
    ca, cb = (a[0] + a[1]) / 2, (b[0] + b[1]) / 2
    b0, b1 = b[0] - cb + ca + wrap(cb - ca), b[1] - cb + ca + wrap(cb - ca)
    return min(a[1], b1) - max(a[0], b0), (a[1] - a[0]), (b1 - b0)


def class_cost(ca, cb):
    """0 same class, 0.25 car/truck/bus confusion, None = impossible pair."""
    ca, cb = int(ca), int(cb)
    if ca == cb:
        return 0.0
    if ca in BIG and cb in BIG:
        return 0.25
    return None


def geo_cost(da, db, cls, baseline=1.5, margin_deg=2.0):
    """Geometric dissimilarity in [0, 1]; None if the two boxes cannot be the same object."""
    ov, la, lb = interval_overlap(da["az"], db["az"])
    depths = [x for x in (da["depth"], db["depth"]) if np.isfinite(x)]
    dep = min(depths) if depths else 8.0
    margin = np.radians(margin_deg) + np.arctan(baseline / max(dep, 3.0))
    if ov + 2 * margin < 0:                                       # bearings do not even touch -> different objects
        return None
    iom = max(ov, 0.0) / max(min(la, lb), 1e-6)
    c_bearing = 1.0 - min(iom, 1.0)
    if da["foot"] is not None and db["foot"] is not None:
        d = float(np.hypot(*(da["foot"] - db["foot"])))
        mean_dep = float(np.mean([da["depth"], db["depth"]]))
        tol = 1.5 + 0.06 * mean_dep + (0.5 * CLS_LEN.get(int(cls), 4.6) if (da["trunc"] or db["trunc"]) else 0.0)
        c_foot = min(d / tol, 2.0) / 2.0
    else:
        c_foot = 0.5                                              # footprint unknown (e.g. cut at the bottom)
    return 0.5 * c_bearing + 0.5 * c_foot


def hsv_hist(img, box, bins=(6, 3, 4)):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    H, W = img.shape[:2]
    x1, x2, y1, y2 = max(0, x1), min(W, x2), max(0, y1), min(H, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    crop = Image.fromarray(img[y1:y2, x1:x2]).convert("HSV").resize((24, 24))
    a = np.asarray(crop, dtype=np.float64) / 256.0
    h, _ = np.histogramdd(a.reshape(-1, 3), bins=bins, range=[(0, 1)] * 3)
    h = h.ravel()
    return h / max(h.sum(), 1e-9)


def app_cost(ha, hb):
    if ha is None or hb is None:
        return 0.5
    bc = float(np.sum(np.sqrt(ha * hb)))
    return float(np.sqrt(max(1.0 - bc, 0.0)))


def _rot(view):
    return view.T[:3, :3]


def warp_b_to_a(va, vb):
    """Homography that maps B pixels into A's image plane (pure rotation, exact for distant content)."""
    R_ab = _rot(va).T @ _rot(vb)                                  # B camera -> A camera
    return va.K @ R_ab @ np.linalg.inv(vb.K)


def keypoint_matrix(va, vb, calibrated=True, nfeat=1500, ratio=0.8, max_shift=0.15):
    """(Na, Nb) integer matrix: number of consistent ORB matches inside box i of A and box j of B."""
    import cv2
    na, nb = len(va.dets), len(vb.dets)
    M = np.zeros((na, nb), dtype=np.int32)
    if na == 0 or nb == 0 or va.img is None or vb.img is None:
        return M
    ga = cv2.cvtColor(va.img, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(vb.img, cv2.COLOR_RGB2GRAY)
    Hm = None
    if calibrated:
        Hm = warp_b_to_a(va, vb)
        gb_w = cv2.warpPerspective(gb, Hm, (va.W, va.H))
    else:
        gb_w = gb
    orb = cv2.ORB_create(nfeat)
    ka, da = orb.detectAndCompute(ga, None)
    kb, db = orb.detectAndCompute(gb_w, None)
    if da is None or db is None or len(ka) < 4 or len(kb) < 4:
        return M
    knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m[0] for m in knn if len(m) == 2 and m[0].distance < ratio * m[1].distance]
    if len(good) < 4:
        return M
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pbw = np.float32([kb[m.trainIdx].pt for m in good])
    if calibrated:
        keep = np.hypot(*(pa - pbw).T) < max_shift * va.W          # after rotation compensation shifts are small
        pa, pbw = pa[keep], pbw[keep]
        pb = cv2.perspectiveTransform(pbw[None], np.linalg.inv(Hm))[0] if len(pbw) else pbw
    else:
        if len(pa) < 8:
            return M
        _, mask = cv2.findFundamentalMat(pa, pbw, cv2.FM_RANSAC, 3.0, 0.99)
        if mask is None:
            return M
        keep = mask.ravel().astype(bool)
        pa, pb = pa[keep], pbw[keep]
    tol = 4.0
    for (xa, ya), (xb, yb) in zip(pa, pb):
        ia = np.where((va.dets[:, 0] - tol <= xa) & (xa <= va.dets[:, 2] + tol) &
                      (va.dets[:, 1] - tol <= ya) & (ya <= va.dets[:, 3] + tol))[0]
        ib = np.where((vb.dets[:, 0] - tol <= xb) & (xb <= vb.dets[:, 2] + tol) &
                      (vb.dets[:, 1] - tol <= yb) & (yb <= vb.dets[:, 3] + tol))[0]
        for i in ia:
            for j in ib:
                M[i, j] += 1
    return M


def kp_cost(n):
    return 0.5 if n == 0 else max(0.0, 0.5 - 0.15 * n)           # no evidence = neutral, >=4 matches = certain


def associate(va, vb, cues=("geo", "app", "kp"), weights=None, thr=DEFAULT_THR, calibrated=True):
    """Match detections of view A with detections of view B.

    Returns dict(matches=[(i, j, cost, {cue: cost})], unmatched_a, unmatched_b, ms={cue: time}).
    """
    w = dict(DEFAULT_W); w.update(weights or {})
    ms = {}
    na, nb = len(va.dets), len(vb.dets)
    out = {"matches": [], "unmatched_a": list(range(na)), "unmatched_b": list(range(nb)), "ms": ms}
    if na == 0 or nb == 0:
        return out

    t = time.perf_counter()
    if "geo" in cues:
        DA = [describe(va, d) for d in va.dets]
        DB = [describe(vb, d) for d in vb.dets]
    else:                                                          # uncalibrated pair (e.g. two phone photos): no geometry
        none = {"foot": None, "depth": np.nan, "az": (0.0, 0.0), "trunc": False, "trunc_b": False}
        DA, DB = [none] * na, [none] * nb
    ms["describe"] = (time.perf_counter() - t) * 1000

    HA = HB = None
    if "app" in cues and va.img is not None and vb.img is not None:
        t = time.perf_counter()
        HA = [hsv_hist(va.img, d[:4]) for d in va.dets]
        HB = [hsv_hist(vb.img, d[:4]) for d in vb.dets]
        ms["app"] = (time.perf_counter() - t) * 1000

    KP = None
    if "kp" in cues and va.img is not None and vb.img is not None:
        try:
            t = time.perf_counter()
            KP = keypoint_matrix(va, vb, calibrated=calibrated)
            ms["kp"] = (time.perf_counter() - t) * 1000
        except ImportError:
            KP = None                                              # no OpenCV on this machine -> cue skipped

    t = time.perf_counter()
    cand = []
    for i in range(na):
        for j in range(nb):
            cc = class_cost(va.dets[i, 5], vb.dets[j, 5])
            if cc is None:
                continue
            parts, total, wsum = {}, 0.0, 0.0
            if "geo" in cues:
                g = geo_cost(DA[i], DB[j], va.dets[i, 5])
                if g is None:
                    continue
                parts["geo"] = g; total += w["geo"] * g; wsum += w["geo"]
            if HA is not None:
                a = app_cost(HA[i], HB[j]); parts["app"] = a; total += w["app"] * a; wsum += w["app"]
            if KP is not None:
                k = kp_cost(int(KP[i, j])); parts["kp"] = k; total += w["kp"] * k; wsum += w["kp"]
            if wsum == 0:
                continue
            cost = total / wsum + cc
            if cost <= thr:
                cand.append((cost, i, j, parts))
    cand.sort(key=lambda x: x[0])
    ua, ub = set(range(na)), set(range(nb))
    for cost, i, j, parts in cand:
        if i in ua and j in ub:
            out["matches"].append((i, j, cost, parts)); ua.discard(i); ub.discard(j)
    out["unmatched_a"], out["unmatched_b"] = sorted(ua), sorted(ub)
    ms["match"] = (time.perf_counter() - t) * 1000
    out["desc"] = (DA, DB)
    return out


def merge_objects(va, vb, res):
    """One entry per real-world object: matched pairs are merged, the rest stay single-view."""
    DA, DB = res["desc"] if "desc" in res else ([describe(va, d) for d in va.dets], [describe(vb, d) for d in vb.dets])
    objs = []
    for i, j, cost, _ in res["matches"]:
        fa, fb = DA[i]["foot"], DB[j]["foot"]
        foot = (fa + fb) / 2 if fa is not None and fb is not None else (fa if fa is not None else fb)
        k = va.dets[i, 5] if va.dets[i, 4] >= vb.dets[j, 4] else vb.dets[j, 5]
        objs.append({"views": "AB", "i": i, "j": j, "cls": int(k), "conf": float(max(va.dets[i, 4], vb.dets[j, 4])),
                     "foot": foot})
    for i in res["unmatched_a"]:
        objs.append({"views": "A", "i": i, "j": None, "cls": int(va.dets[i, 5]), "conf": float(va.dets[i, 4]), "foot": DA[i]["foot"]})
    for j in res["unmatched_b"]:
        objs.append({"views": "B", "i": None, "j": j, "cls": int(vb.dets[j, 5]), "conf": float(vb.dets[j, 4]), "foot": DB[j]["foot"]})
    return objs


def perturb_pose(view, rng, sigma_m, sigma_deg):
    """Simulate imperfect V2X localisation: noisy relative pose of the other camera."""
    if sigma_m == 0 and sigma_deg == 0:
        return view
    T = view.T.copy()
    a = np.radians(rng.normal(0, sigma_deg))
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
    T[:3, :3] = R @ T[:3, :3]
    T[:2, 3] += rng.normal(0, sigma_m, 2)
    return View(view.K, T, view.dets, (view.W, view.H), view.img, view.name)


def message_bytes(dets_a):
    """Size of what a vehicle would send: per detection class(1) + box(4x2 B) + conf(1) = 10 B, plus a 4 B header."""
    return 4 + 10 * len(dets_a)

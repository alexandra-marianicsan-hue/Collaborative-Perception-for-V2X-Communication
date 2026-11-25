"""Prepare nuScenes for the collaborative-perception project.

Real images from neighbouring cameras of the ego car (they share ~15-20 degrees of view, so a vehicle at
the seam appears cut in half: one part in each image) play the role of "vehicle A" and "vehicle B".
nuScenes has no 2D boxes, so they are derived by projecting the annotated 3D boxes. The annotation
`instance_token` tells us which boxes in the two images are the SAME vehicle (association ground truth).

Output (under --out):
  pack/            pairs.json + images/ : evaluation set for the board (eval_assoc.py / demo_assoc.py)
  yolo/            images/{train,val}, labels/{train,val}, nuscenes.yaml : fine-tuning set (single-camera labels)

  python prepare_nuscenes.py --dataroot D:/nuscenes --version v1.0-mini --out data/nuscenes_v2x
Needs: pip install nuscenes-devkit pillow numpy
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from nusc_geometry import PAIRS, box_corners, project_box

CLASS_MAP = {"vehicle.bicycle": 0, "vehicle.car": 1, "vehicle.motorcycle": 2,
             "vehicle.bus.bendy": 3, "vehicle.bus.rigid": 3, "vehicle.truck": 4}
CLASS_NAMES = ["bicycle", "car", "motorcycle", "bus", "truck"]
CAMS = sorted({c for p in PAIRS for c in p})


def tf(trans, rot):
    from pyquaternion import Quaternion
    T = np.eye(4)
    T[:3, :3] = Quaternion(rot).rotation_matrix
    T[:3, 3] = trans
    return T


def cam_record(nusc, sample, cam, width, args, inst_ids):
    """Everything we need about one camera image of one sample."""
    sd = nusc.get("sample_data", sample["data"][cam])
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ep = nusc.get("ego_pose", sd["ego_pose_token"])
    W0, H0 = sd["width"], sd["height"]
    scale = width / W0
    W, H = width, int(round(H0 * scale))
    K = np.array(cs["camera_intrinsic"], dtype=np.float64)
    T_e_c = tf(cs["translation"], cs["rotation"])
    T_g_e = tf(ep["translation"], ep["rotation"])
    T_g_c = T_g_e @ T_e_c
    inv = np.linalg.inv(T_g_c)
    objs = []
    from pyquaternion import Quaternion
    for tok in sample["anns"]:
        ann = nusc.get("sample_annotation", tok)
        inst = nusc.get("instance", ann["instance_token"])
        name = nusc.get("category", inst["category_token"])["name"]
        if name not in CLASS_MAP:
            continue
        if int(ann["visibility_token"]) < args.min_vis or ann["num_lidar_pts"] + ann["num_radar_pts"] <= 0:
            continue
        local = box_corners((0, 0, 0), ann["size"], 0.0)
        cg = Quaternion(ann["rotation"]).rotation_matrix @ local + np.array(ann["translation"]).reshape(3, 1)
        cc = inv[:3, :3] @ cg + inv[:3, 3:4]
        pr = project_box(cc, K, W0, H0)
        if pr is None or pr["depth"] > args.max_depth:
            continue
        frac = pr["clip_area"] / max(pr["full_area"], 1e-6)
        b = [v * scale for v in pr["box"]]
        if frac < args.min_frac or (b[2] - b[0]) < args.min_px or (b[3] - b[1]) < args.min_px:
            continue
        iid = inst_ids.setdefault(ann["instance_token"], len(inst_ids))
        objs.append({"inst": iid, "cls": CLASS_MAP[name], "box": [round(v, 1) for v in b], "frac": round(frac, 3),
                     "depth": round(pr["depth"], 1), "len": float(ann["size"][1])})
    K2 = K.copy(); K2[:2] *= scale
    return {"sd": sd, "K": K2, "T_g_c": T_g_c, "T_g_e": T_g_e, "T_e_c": T_e_c, "objs": objs, "size": (W, H)}


def save_image(nusc, rec, dst, quality=85):
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(Path(nusc.dataroot) / rec["sd"]["filename"]).convert("RGB").resize(rec["size"], Image.BILINEAR)
    img.save(dst, quality=quality)


def yolo_label(rec):
    W, H = rec["size"]
    lines = []
    for o in rec["objs"]:
        x1, y1, x2, y2 = o["box"]
        lines.append(f"{o['cls']} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--out", default="data/nuscenes_v2x")
    ap.add_argument("--width", type=int, default=800, help="saved image width (16:9)")
    ap.add_argument("--pack-split", default="val", choices=["train", "val", "all"])
    ap.add_argument("--max-pairs", type=int, default=150, help="pairs with >=1 shared vehicle kept in the pack")
    ap.add_argument("--neg-frac", type=float, default=0.3, help="extra pairs without shared vehicles, as fraction of positives")
    ap.add_argument("--min-vis", type=int, default=2, help="nuScenes visibility bin (1-4)")
    ap.add_argument("--min-frac", type=float, default=0.3, help="min fraction of the 3D box visible in the image")
    ap.add_argument("--min-px", type=float, default=10, help="min 2D box side in saved-image pixels")
    ap.add_argument("--max-depth", type=float, default=70.0)
    ap.add_argument("--no-yolo", action="store_true")
    args = ap.parse_args()

    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    splits = create_splits_scenes()
    key = "mini" if "mini" in args.version else None
    train_names = set(splits["mini_train" if key else "train"]); val_names = set(splits["mini_val" if key else "val"])
    out = Path(args.out); pack = out / "pack"; yolo = out / "yolo"
    inst_ids = {}
    pairs, cls_count = [], Counter()
    n_img = 0

    for scene in nusc.scene:
        split = "train" if scene["name"] in train_names else "val" if scene["name"] in val_names else None
        if split is None:
            continue
        tok = scene["first_sample_token"]
        scene_pairs = []
        while tok:
            sample = nusc.get("sample", tok)
            recs = {c: cam_record(nusc, sample, c, args.width, args, inst_ids) for c in CAMS}
            if not args.no_yolo:
                for c, r in recs.items():
                    stem = f"{sample['token'][:12]}_{c}"
                    save_image(nusc, r, yolo / "images" / split / f"{stem}.jpg")
                    lp = yolo / "labels" / split / f"{stem}.txt"
                    lp.parent.mkdir(parents=True, exist_ok=True)
                    lp.write_text(yolo_label(r))
                    cls_count.update((split, o["cls"]) for o in r["objs"]); n_img += 1
            if args.pack_split in (split, "all"):
                for ca, cb in PAIRS:
                    ra, rb = recs[ca], recs[cb]
                    inst_a = {o["inst"] for o in ra["objs"]}; inst_b = {o["inst"] for o in rb["objs"]}
                    shared = inst_a & inst_b
                    T_ref_b = np.linalg.inv(ra["T_g_e"]) @ rb["T_g_c"]         # camera B in the ego frame of A's timestamp
                    scene_pairs.append({
                        "sample": sample["token"], "scene": scene["name"], "split": split, "camA": ca, "camB": cb,
                        "imgA": f"images/{sample['token'][:12]}_{ca}.jpg", "imgB": f"images/{sample['token'][:12]}_{cb}.jpg",
                        "size": list(ra["size"]), "KA": ra["K"].tolist(), "KB": rb["K"].tolist(),
                        "TA": ra["T_e_c"].tolist(), "TB": T_ref_b.tolist(),
                        "objsA": ra["objs"], "objsB": rb["objs"], "n_shared": len(shared),
                        "_recs": (ra, rb)})
            tok = sample["next"]
        pairs += scene_pairs

    # choose pack pairs: all positives (evenly subsampled) + some negatives
    pos = [p for p in pairs if p["n_shared"] > 0]
    neg = [p for p in pairs if p["n_shared"] == 0]
    if len(pos) > args.max_pairs:
        pos = [pos[i] for i in np.linspace(0, len(pos) - 1, args.max_pairs).astype(int)]
    rng = np.random.default_rng(0)
    n_neg = min(len(neg), int(round(args.neg_frac * len(pos))))
    neg = [neg[i] for i in sorted(rng.choice(len(neg), n_neg, replace=False))] if n_neg else []
    chosen = sorted(pos + neg, key=lambda p: (p["scene"], p["sample"], p["camA"]))
    for p in chosen:
        ra, rb = p.pop("_recs")
        save_image(nusc, ra, pack / p["imgA"]); save_image(nusc, rb, pack / p["imgB"])
    if chosen:
        pack.mkdir(parents=True, exist_ok=True)
        (pack / "pairs.json").write_text(json.dumps({"classes": CLASS_NAMES, "pairs": chosen}))

    if not args.no_yolo:
        (yolo / "nuscenes.yaml").write_text("train: images/train\nval: images/val\nnc: 5\nnames: " + json.dumps(CLASS_NAMES) + "\n")
    print(f"\nYOLO set : {n_img} images in {yolo}")
    for sp in ("train", "val"):
        print(f"  {sp:5s} boxes per class:", {CLASS_NAMES[c]: n for (s, c), n in sorted(cls_count.items()) if s == sp})
    n_trunc = sum(1 for p in chosen for o in p["objsA"] + p["objsB"] if o["frac"] < 0.9)
    shared_total = sum(p["n_shared"] for p in chosen)
    print(f"Pack     : {len(chosen)} pairs in {pack} ({len(pos)} with shared vehicles, {len(neg)} without) | "
          f"{shared_total} shared vehicle instances | {n_trunc} partial (cut) boxes")


if __name__ == "__main__":
    main()

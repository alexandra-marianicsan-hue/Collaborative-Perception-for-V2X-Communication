"""Generate a tiny FAKE nuScenes database (v1.0-mini layout) to dry-run the whole pipeline without downloading
the 4 GB dataset: random vehicles seen by the six nuScenes-like cameras, 2 scenes x 5 samples.
Images are flat-colour boxes on noise - only meant to test code paths, never to measure accuracy.

  python make_fake_nuscenes.py --out data/fake_nuscenes
  python prepare_nuscenes.py --dataroot data/fake_nuscenes --out data/fake_out
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pyquaternion import Quaternion

from nusc_geometry import RIG, box_corners, cam_to_ego, intrinsics, project_box
from test_assoc_synth import SIZES, make_world

CATS = {1: "vehicle.car", 3: "vehicle.bus.rigid", 4: "vehicle.truck", 2: "vehicle.motorcycle", 0: "vehicle.bicycle"}
W, H = 1600, 900


def tok(prefix, i):
    return f"{prefix}{i:028d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/fake_nuscenes")
    ap.add_argument("--samples", type=int, default=5)
    a = ap.parse_args()
    root = Path(a.out); tdir = root / "v1.0-mini"; tdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)

    cat_tok = {c: tok("cat", c) for c in list(CATS) + [9]}
    tables = {k: [] for k in ["category", "attribute", "visibility", "instance", "sensor", "calibrated_sensor", "ego_pose",
                              "log", "scene", "sample", "sample_data", "sample_annotation", "map"]}
    for c, n in CATS.items():
        tables["category"].append({"token": cat_tok[c], "name": n, "description": ""})
    tables["category"].append({"token": cat_tok[9], "name": "human.pedestrian.adult", "description": ""})
    tables["attribute"].append({"token": "attr0", "name": "vehicle.parked", "description": ""})
    for v in "1234":
        tables["visibility"].append({"token": v, "level": f"v{v}", "description": ""})
    cams = list(RIG)
    cs_tok = {}
    for i, cam in enumerate(cams):
        x, y, z, yaw, fx = RIG[cam]
        T = cam_to_ego(x, y, z, yaw)
        tables["sensor"].append({"token": tok("sen", i), "channel": cam, "modality": "camera"})
        cs_tok[cam] = tok("cs", i)
        tables["calibrated_sensor"].append({
            "token": cs_tok[cam], "sensor_token": tok("sen", i), "translation": [x, y, z],
            "rotation": list(Quaternion(matrix=T[:3, :3], atol=1e-6).elements),
            "camera_intrinsic": intrinsics(fx, W, H).tolist()})
    tables["log"].append({"token": "log0", "logfile": "fake", "vehicle": "v", "date_captured": "2026", "location": "x"})
    tables["map"].append({"token": "map0", "log_tokens": ["log0"], "category": "semantic_prior", "filename": "maps/none.png"})

    inst_n = ann_n = sd_n = ep_n = 0
    for sc_i, sc_name in enumerate(["scene-0061", "scene-0103"]):       # one mini_train, one mini_val scene
        objs = make_world(rng, n=16)
        yaw0 = 0.3 + sc_i
        base = np.array([100.0 + 40 * sc_i, 50.0])
        R0 = Quaternion(axis=[0, 0, 1], angle=yaw0).rotation_matrix
        inst_tok = {}
        for o in objs:
            inst_tok[o["id"]] = tok("ins", inst_n); inst_n += 1
            tables["instance"].append({"token": inst_tok[o["id"]], "category_token": cat_tok[o["cls"]],
                                       "nbr_annotations": a.samples, "first_annotation_token": "", "last_annotation_token": ""})
        s_toks = [tok("smp", sc_i * 100 + k) for k in range(a.samples)]
        tables["scene"].append({"token": tok("sce", sc_i), "name": sc_name, "description": "", "log_token": "log0",
                                "nbr_samples": a.samples, "first_sample_token": s_toks[0], "last_sample_token": s_toks[-1]})
        for k, st in enumerate(s_toks):
            tables["sample"].append({"token": st, "timestamp": k * 500000, "scene_token": tok("sce", sc_i),
                                     "prev": s_toks[k - 1] if k else "", "next": s_toks[k + 1] if k < a.samples - 1 else ""})
            for o in objs:                                              # objects are static in the global frame
                c = R0 @ np.array(o["c"]) + np.append(base, 0.0)
                tables["sample_annotation"].append({
                    "token": tok("ann", ann_n), "sample_token": st, "instance_token": inst_tok[o["id"]],
                    "visibility_token": "4", "attribute_tokens": [], "translation": c.tolist(),
                    "size": list(o["wlh"]), "rotation": list(Quaternion(axis=[0, 0, 1], angle=o["yaw"] + yaw0).elements),
                    "prev": "", "next": "", "num_lidar_pts": 50, "num_radar_pts": 0})
                ann_n += 1
            for ci, cam in enumerate(cams):
                ego_xy = base + R0[:2, 0] * (1.5 * k + 0.3 * ci)         # ego keeps moving between camera exposures
                ep_tok = tok("ego", ep_n); ep_n += 1
                tables["ego_pose"].append({"token": ep_tok, "timestamp": k * 500000 + ci * 4000,
                                           "rotation": list(Quaternion(axis=[0, 0, 1], angle=yaw0).elements),
                                           "translation": [float(ego_xy[0]), float(ego_xy[1]), 0.0]})
                fn = f"samples/{cam}/{sc_name}_{k}_{cam}.jpg"
                tables["sample_data"].append({
                    "token": tok("sdt", sd_n), "sample_token": st, "ego_pose_token": ep_tok,
                    "calibrated_sensor_token": cs_tok[cam], "timestamp": k * 500000 + ci * 4000, "fileformat": "jpg",
                    "is_key_frame": True, "height": H, "width": W, "filename": fn, "prev": "", "next": ""})
                sd_n += 1
                # image: noise + one flat-coloured rectangle per projected object (colour = identity)
                img = Image.fromarray((rng.random((H // 4, W // 4, 3)) * 120).astype(np.uint8)).resize((W, H))
                dr = ImageDraw.Draw(img)
                T = cam_to_ego(*RIG[cam][:4]); K = intrinsics(RIG[cam][4], W, H)
                Tg_e = np.eye(4); Tg_e[:3, :3] = R0; Tg_e[:3, 3] = [ego_xy[0], ego_xy[1], 0.0]
                inv = np.linalg.inv(Tg_e @ T)
                for o in objs:
                    cg = R0 @ box_corners(o["c"], o["wlh"], o["yaw"]) + np.append(base, 0.0).reshape(3, 1)
                    pr = project_box(inv[:3, :3] @ cg + inv[:3, 3:4], K, W, H)
                    if pr:
                        col = tuple(int(v) for v in np.random.default_rng(o["id"] + 10 * sc_i).integers(80, 256, 3))
                        dr.rectangle(pr["box"], fill=col)
                (root / fn).parent.mkdir(parents=True, exist_ok=True)
                img.save(root / fn, quality=80)
    (root / "maps").mkdir(exist_ok=True); Image.new("L", (8, 8)).save(root / "maps/none.png")   # devkit insists on a map file
    for name, rows in tables.items():
        (tdir / f"{name}.json").write_text(json.dumps(rows))
    print(f"fake nuScenes written to {root}: {len(tables['sample'])} samples, {sd_n} images, {ann_n} annotations")


if __name__ == "__main__":
    main()

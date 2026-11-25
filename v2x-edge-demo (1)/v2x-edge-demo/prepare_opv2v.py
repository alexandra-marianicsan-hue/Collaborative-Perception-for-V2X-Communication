"""Build a YOLO dataset + collaboration index from a (small) part of OPV2V.

Expected layout (as downloaded):  <root>/<scenario>/<cav_id>/<ts>.yaml, <ts>_camera0.png
Use the smallest split you can download (e.g. test_culvercity or a couple of scenarios).

2D labels are NOT given by OPV2V: they are obtained by projecting the 3D vehicle boxes
(from the yaml) into the front camera (single class 'vehicle'). Run verify_labels.py to check.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from opv2v_geometry import box2d, vehicle_center_world

try:
    Loader = yaml.CSafeLoader
except AttributeError:
    Loader = yaml.SafeLoader


def load_yaml(p):
    with open(p) as f:
        return yaml.load(f, Loader=Loader)


def list_cavs(sdir):
    return sorted((d for d in sdir.iterdir() if d.is_dir() and d.name.isdigit()), key=lambda d: int(d.name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="OPV2V split folder, e.g. opv2v/test_culvercity")
    ap.add_argument("--out", default="data/opv2v")
    ap.add_argument("--stride", type=int, default=4, help="keep every N-th timestamp")
    ap.add_argument("--max-scenarios", type=int, default=6)
    ap.add_argument("--max-cavs", type=int, default=4)
    ap.add_argument("--max-val-ts", type=int, default=60)
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    scens = sorted(d for d in root.iterdir() if d.is_dir())[: args.max_scenarios]
    if not scens:
        raise SystemExit(f"No scenario folders in {root}")

    # ---- list (scenario, ts) and decide the split -------------------------------------
    entries = []
    for si, s in enumerate(scens):
        cavs = list_cavs(s)[: args.max_cavs]
        if len(cavs) < 2:
            continue
        ts_all = sorted(p.stem for p in cavs[0].glob("*.yaml") if p.stem.isdigit())[:: args.stride]
        for ti, ts in enumerate(ts_all):
            if all((c / f"{ts}.yaml").exists() and (c / f"{ts}_camera0.png").exists() for c in cavs):
                if len(scens) >= 4:
                    split = "val" if si >= len(scens) - max(1, len(scens) // 4) else "train"
                else:
                    split = "val" if ti >= int(0.75 * len(ts_all)) else "train"
                entries.append((s, cavs, ts, split))
    val_idx = [i for i, e in enumerate(entries) if e[3] == "val"]
    if len(val_idx) > args.max_val_ts:
        keep = set(np.array(val_idx)[np.linspace(0, len(val_idx) - 1, args.max_val_ts).astype(int)].tolist())
        entries = [e for i, e in enumerate(entries) if e[3] == "train" or i in keep]

    for sp in ("train", "val"):
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)

    index, n_img, n_box, ground = [], {"train": 0, "val": 0}, {"train": 0, "val": 0}, []
    for s, cavs, ts, split in entries:
        gt_world, cav_entries = {}, []
        for c in cavs:
            d = load_yaml(c / f"{ts}.yaml")
            pose, K = d["camera0"]["cords"], np.array(d["camera0"]["intrinsic"], float)
            img = cv2.imread(str(c / f"{ts}_camera0.png"))
            H, W = img.shape[:2]
            boxes = []
            for vid, veh in (d.get("vehicles") or {}).items():
                ctr = vehicle_center_world(veh)
                gt_world[int(vid)] = [float(ctr[0]), float(ctr[1])]
                ground.append(veh["location"][2] - pose[2])
                b = box2d(veh, pose, K, W, H)
                if b:
                    boxes.append(b)
            name = f"{s.name}__{c.name}__{ts}"
            cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            lines = [f"0 {(b[0] + b[2]) / 2 / W:.6f} {(b[1] + b[3]) / 2 / H:.6f} "
                     f"{(b[2] - b[0]) / W:.6f} {(b[3] - b[1]) / H:.6f}" for b in boxes]
            (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            n_img[split] += 1
            n_box[split] += len(boxes)
            cav_entries.append(dict(cav=int(c.name), image=f"images/{split}/{name}.jpg", pose=pose,
                                    K=K.tolist(), gt2d=boxes, W=W, H=H))
        if split == "val":
            index.append(dict(scenario=s.name, ts=ts, ego=int(cavs[0].name), cavs=cav_entries,
                              gt_world=[dict(id=k, x=v[0], y=v[1]) for k, v in gt_world.items()]))

    g = float(np.median(ground)) if ground else -1.0
    json.dump(dict(root=str(out), ground_offset=g, timestamps=index), open(out.parent / "opv2v_index.json", "w"))
    (out.parent / "opv2v.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: vehicle\n")
    md = ["# OPV2V subset statistics", "",
          f"Source: OPV2V (CARLA, V2V), split `{root.name}`, {len(scens)} scenario(s), stride {args.stride}, front camera (camera0).",
          "2D boxes = 3D vehicle boxes projected into the camera (single class `vehicle`).", "",
          f"- train: {n_img['train']} images, {n_box['train']} boxes", f"- val: {n_img['val']} images, {n_box['val']} boxes "
          f"({len(index)} timestamps with 2-{args.max_cavs} CAVs)", f"- median (vehicle ground z - camera z): {g:.2f} m"]
    (out.parent / "OPV2V_STATS.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

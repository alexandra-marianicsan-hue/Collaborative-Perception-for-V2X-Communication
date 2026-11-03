"""Build a small vehicle-detection dataset (YOLO format) from COCO128.

Keeps only vehicle classes (bicycle, car, motorcycle, bus, truck) and images
containing at least one of them. Writes data/vehicles/ + data/vehicles.yaml
+ data/DATASET_STATS.md (paste into the README / slides).
"""
import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

from ultralytics.utils.downloads import download

URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
COCO_TO_VEH = {1: 0, 2: 1, 3: 2, 5: 3, 7: 4}
NAMES = ["bicycle", "car", "motorcycle", "bus", "truck"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path("data")
    root.mkdir(exist_ok=True)
    src = root / "coco128"
    if not src.exists():
        download(URL, dir=root, unzip=True, delete=True)
    img_dir = next(src.glob("images/*"))
    lbl_dir = next(src.glob("labels/*"))

    items = []
    for lp in sorted(lbl_dir.glob("*.txt")):
        rows = []
        for line in lp.read_text().splitlines():
            p = line.split()
            if len(p) >= 5 and int(float(p[0])) in COCO_TO_VEH:
                rows.append(" ".join([str(COCO_TO_VEH[int(float(p[0]))])] + p[1:5]))
        ip = img_dir / (lp.stem + ".jpg")
        if rows and ip.exists():
            items.append((ip, rows))

    random.Random(args.seed).shuffle(items)
    n_val = max(1, int(len(items) * args.val_ratio))
    out = root / "vehicles"
    if out.exists():
        shutil.rmtree(out)
    counts = {"train": Counter(), "val": Counter()}
    n_img = Counter()
    for i, (ip, rows) in enumerate(items):
        split = "val" if i < n_val else "train"
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        shutil.copy(ip, out / "images" / split / ip.name)
        (out / "labels" / split / (ip.stem + ".txt")).write_text("\n".join(rows) + "\n")
        n_img[split] += 1
        counts[split].update(int(r.split()[0]) for r in rows)

    yaml = f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n"
    yaml += "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES))
    (root / "vehicles.yaml").write_text(yaml)

    md = ["# Dataset statistics", "", "Source: COCO128 (public, 128 COCO images), filtered to vehicle classes.", "",
          f"- train images: {n_img['train']}, val images: {n_img['val']}", "",
          "| class | train boxes | val boxes |", "|---|---|---|"]
    md += [f"| {n} | {counts['train'][i]} | {counts['val'][i]} |" for i, n in enumerate(NAMES)]
    (root / "DATASET_STATS.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

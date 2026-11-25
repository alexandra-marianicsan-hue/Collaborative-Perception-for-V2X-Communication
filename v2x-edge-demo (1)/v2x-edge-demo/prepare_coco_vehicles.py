"""Build the vehicle dataset used for the final fine-tuning (COCO val2017 -> 5 vehicle classes, YOLO format).

  python prepare_coco_vehicles.py            # downloads val2017 (~1 GB) + annotations (~250 MB)
Output: data/vehicles_big/{images,labels}/{train,val}, data/vehicles_big.yaml, data/DATASET_STATS.md
COCO val2017 is not part of the pretraining set of yolov8n.pt (trained on train2017).
"""
import argparse
import json
import random
import shutil
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

CAT = {2: 0, 3: 1, 4: 2, 6: 3, 8: 4}      # COCO category ids: bicycle, car, motorcycle, bus, truck
NAMES = ["bicycle", "car", "motorcycle", "bus", "truck"]
URLS = {"val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
        "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"}


def fetch(name):
    if not Path(name).exists():
        print("downloading", name)
        urllib.request.urlretrieve(URLS[name], name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-images", type=int, default=1500)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--min-box", type=float, default=8, help="min box side in px (before resize)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    for n in URLS:
        fetch(n)
    if not Path("val2017").exists():
        zipfile.ZipFile("val2017.zip").extractall(".")
    ann = Path("annotations/instances_val2017.json")
    if not ann.exists():
        zipfile.ZipFile("annotations_trainval2017.zip").extract("annotations/instances_val2017.json", ".")
    a = json.load(open(ann))
    imgs = {i["id"]: i for i in a["images"]}
    boxes = {}
    for x in a["annotations"]:
        if x["category_id"] in CAT and not x.get("iscrowd") and x["bbox"][2] > args.min_box and x["bbox"][3] > args.min_box:
            boxes.setdefault(x["image_id"], []).append(x)
    ids = sorted(boxes)
    random.Random(args.seed).shuffle(ids)
    ids = ids[: args.max_images]

    out = Path("data/vehicles_big")
    shutil.rmtree(out, ignore_errors=True)
    nval = int(args.val_ratio * len(ids))
    cnt = {"train": Counter(), "val": Counter()}
    n_img = Counter()
    for k, iid in enumerate(ids):
        sp = "val" if k < nval else "train"
        im = imgs[iid]; W, H = im["width"], im["height"]
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)
        shutil.copy(f"val2017/{im['file_name']}", out / "images" / sp / im["file_name"])
        lines = []
        for b in boxes[iid]:
            x, y, w, h = b["bbox"]
            lines.append(f"{CAT[b['category_id']]} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
            cnt[sp][CAT[b["category_id"]]] += 1
        (out / "labels" / sp / (Path(im["file_name"]).stem + ".txt")).write_text("\n".join(lines) + "\n")
        n_img[sp] += 1

    Path("data/vehicles_big.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
    md = ["# Dataset statistics (vehicles_big)", "",
          "Source: COCO val2017 (public), images containing bicycle/car/motorcycle/bus/truck, random subset (seed 0).", "",
          f"- train images: {n_img['train']}, val images: {n_img['val']}", "",
          "| class | train boxes | val boxes |", "|---|---|---|"]
    md += [f"| {n} | {cnt['train'][i]} | {cnt['val'][i]} |" for i, n in enumerate(NAMES)]
    Path("data/DATASET_STATS.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

"""Transfer learning: fine-tune a COCO-pretrained YOLOv8n on the vehicle dataset.

Backbone layers are frozen (--freeze 10) -> classic transfer learning, fast on CPU.
Use --data kitti.yaml --fraction 0.2 for a bigger real driving dataset (auto-download).
"""
import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--data", default="data/vehicles.yaml")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--freeze", type=int, default=10, help="number of frozen layers (backbone)")
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction of train set to use")
    ap.add_argument("--device", default=None, help="'cpu', '0', ... (default: auto)")
    args = ap.parse_args()

    model = YOLO(args.weights)
    res = model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                      freeze=args.freeze, fraction=args.fraction, device=args.device,
                      project="runs", name="finetune", exist_ok=True, workers=2, plots=True)
    Path("models").mkdir(exist_ok=True)
    shutil.copy(model.trainer.best, "models/best.pt")
    Path("results").mkdir(exist_ok=True)
    try:
        json.dump({k: float(v) for k, v in res.results_dict.items()},
                  open("results/train_metrics.json", "w"), indent=2)
    except Exception:
        pass
    print("Saved models/best.pt")


if __name__ == "__main__":
    main()

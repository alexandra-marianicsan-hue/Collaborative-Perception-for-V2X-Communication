"""Vehicle detector wrapper with a per-stage timing breakdown (used by eval_assoc.py and demo_assoc.py).

Output classes are always 0 bicycle, 1 car, 2 motorcycle, 3 bus, 4 truck (the fine-tuned model's order);
a COCO-pretrained 80-class model is remapped automatically.
"""
import time

import numpy as np
from PIL import Image

from tflite_utils import TfliteDetector

COCO_TO_5 = {1: 0, 2: 1, 3: 2, 5: 3, 7: 4}


def letterbox(img, S):
    w, h = img.size
    r = min(S / w, S / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    canvas = Image.new("RGB", (S, S), (114, 114, 114))
    left, top = (S - nw) // 2, (S - nh) // 2
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (left, top))
    return np.asarray(canvas), r, left, top


class VehicleDetector:
    def __init__(self, model, delegate=None, threads=4):
        self.det = TfliteDetector(model, delegate, threads)
        shp = self.det.it.get_output_details()[0]["shape"]
        self.nc = int(min(shp[1:])) - 4
        self.det.run(np.zeros((self.det.imgsz, self.det.imgsz, 3), np.uint8))      # warm-up (NPU graph compile)

    def __call__(self, img, conf=0.25):
        """img: PIL RGB. Returns (N,6) detections in image pixels and {stage: ms}."""
        t0 = time.perf_counter()
        arr, r, left, top = letterbox(img, self.det.imgsz)
        t1 = time.perf_counter()
        out, pre_ms, inv_ms = self.det.run(arr)                                    # quantise + invoke (+ dequantise)
        t2 = time.perf_counter()
        d = self.det.decode(out, conf)
        if len(d):
            if self.nc != 5:
                keep = np.isin(d[:, 5], list(COCO_TO_5))
                d = d[keep]
                d[:, 5] = [COCO_TO_5[int(k)] for k in d[:, 5]]
            d[:, [0, 2]] = ((d[:, [0, 2]] - left) / r).clip(0, img.size[0])
            d[:, [1, 3]] = ((d[:, [1, 3]] - top) / r).clip(0, img.size[1])
        t3 = time.perf_counter()
        run_ms = (t2 - t1) * 1e3
        return d, {"letterbox": (t1 - t0) * 1e3, "quantize": pre_ms, "invoke": inv_ms,
                   "decode": (t3 - t2) * 1e3 + max(run_ms - pre_ms - inv_ms, 0.0)}   # decode = dequantise + decode + NMS + rescale

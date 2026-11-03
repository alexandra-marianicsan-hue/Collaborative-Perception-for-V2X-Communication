"""Minimal ONNX Runtime YOLOv8 detector (no PyTorch needed on the edge device)."""
import ast

import cv2
import numpy as np
import onnxruntime as ort

from collab import nms


def letterbox(img, size):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


class OnnxDetector:
    def __init__(self, model_path, conf=0.25, iou=0.45, providers=None):
        avail = ort.get_available_providers()
        if providers is None:
            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.providers = self.session.get_providers()
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.imgsz = int(inp.shape[2]) if isinstance(inp.shape[2], int) else 640
        meta = self.session.get_modelmeta().custom_metadata_map
        self.names = {int(k): v for k, v in ast.literal_eval(meta["names"]).items()} if "names" in meta else {}
        self.conf, self.iou = conf, iou

    def __call__(self, bgr):
        img, r, left, top = letterbox(bgr, self.imgsz)
        blob = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        out = self.session.run(None, {self.input_name: blob})[0][0].T      # (N, 4+nc)
        scores = out[:, 4:]
        cls, conf = scores.argmax(1), scores.max(1)
        m = conf >= self.conf
        out, cls, conf = out[m], cls[m], conf[m]
        if not len(out):
            return np.zeros((0, 6), np.float32)
        cx, cy, w, h = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / r
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / r
        H, W = bgr.shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)
        keep = nms(boxes, conf, cls, self.iou)
        return np.concatenate([boxes[keep], conf[keep, None], cls[keep, None]], 1).astype(np.float32)

"""TFLite detector for the edge board (numpy + tflite_runtime only; no OpenCV needed).

Works with FP32/FP16/INT8 YOLOv8 TFLite models exported by Ultralytics. With
delegate='/usr/lib/libvx_delegate.so' the supported ops run on the i.MX 8M Plus NPU.
"""
import os
import time

import numpy as np

from collab import nms


def load_interpreter(model_path, delegate=None, threads=4):
    try:
        from tflite_runtime.interpreter import Interpreter, load_delegate
    except ImportError:
        try:
            from ai_edge_litert.interpreter import Interpreter, load_delegate
        except ImportError:
            import tensorflow as tf
            Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate
    dels = [load_delegate(delegate)] if delegate else None
    kwargs = {}
    if os.environ.get("NO_XNNPACK"):          # workaround: some quantized models fail in the XNNPACK delegate
        try:
            from tflite_runtime.interpreter import OpResolverType
            kwargs["experimental_op_resolver_type"] = OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        except ImportError:
            pass
    return Interpreter(model_path=model_path, num_threads=threads, experimental_delegates=dels, **kwargs)


class TfliteDetector:
    def __init__(self, model_path, delegate=None, threads=4):
        t = time.perf_counter()
        self.it = load_interpreter(model_path, delegate, threads)
        self.it.allocate_tensors()
        self.load_ms = (time.perf_counter() - t) * 1000
        i, o = self.it.get_input_details()[0], self.it.get_output_details()[0]
        self.in_idx, self.out_idx = i["index"], o["index"]
        self.in_dtype, self.in_q = i["dtype"], i["quantization"]
        self.out_dtype, self.out_q = o["dtype"], o["quantization"]
        shape = i["shape"]
        if len(shape) == 4 and shape[3] == 3:          # NHWC (onnx2tf / legacy exports)
            self.layout, self.imgsz = "NHWC", int(shape[1])
        elif len(shape) == 4 and shape[1] == 3:        # NCHW (Ultralytics LiteRT export)
            self.layout, self.imgsz = "NCHW", int(shape[2])
        else:
            raise ValueError(f"Unsupported input shape {shape}")

    def preprocess(self, img_u8):
        if self.in_dtype != np.float32 and not os.environ.get("V2X_SLOW_PRE"):
            scale, zp = self.in_q
            if abs(scale * 255.0 - 1.0) < 1e-3:          # Ultralytics int8 input: q = pixel + zero_point, no float maths needed
                x = img_u8.transpose(2, 0, 1) if self.layout == "NCHW" else img_u8
                info = np.iinfo(self.in_dtype)
                q = np.clip(x.astype(np.int16) + int(round(zp)), info.min, info.max)
                return np.ascontiguousarray(q.astype(self.in_dtype))[None]
        x = img_u8.astype(np.float32) / 255.0
        if self.layout == "NCHW":
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
        if self.in_dtype == np.float32:
            return x[None]
        scale, zp = self.in_q
        info = np.iinfo(self.in_dtype)
        return np.clip(np.round(x / scale + zp), info.min, info.max).astype(self.in_dtype)[None]

    def run(self, img_u8):
        """img_u8: letterboxed RGB uint8 (S,S,3). Returns raw output, preprocess ms, invoke ms."""
        t0 = time.perf_counter()
        x = self.preprocess(img_u8)
        t1 = time.perf_counter()
        self.it.set_tensor(self.in_idx, x)
        self.it.invoke()
        out = self.it.get_tensor(self.out_idx)
        t2 = time.perf_counter()
        if self.out_dtype != np.float32:
            scale, zp = self.out_q
            out = (out.astype(np.float32) - zp) * scale
        return out, (t1 - t0) * 1000, (t2 - t1) * 1000

    def decode(self, out, conf=0.25, iou=0.45, topk=300):
        """Raw output -> (N,6) x1,y1,x2,y2,conf,cls in letterboxed pixel coordinates."""
        out = out[0] if out.ndim == 3 else out
        if out.shape[0] < out.shape[1]:
            out = out.T                                      # (N, 4+nc)
        boxes = out[:, :4].astype(np.float32)
        if boxes.size and boxes.max() <= 2.0:                # TFLite exports use normalised xywh
            boxes = boxes * self.imgsz
        scores = out[:, 4:]
        cls, sc = scores.argmax(1), scores.max(1)
        m = sc >= conf
        boxes, cls, sc = boxes[m], cls[m], sc[m]
        if not len(sc):
            return np.zeros((0, 6), np.float32)
        if len(sc) > topk:
            o = np.argsort(-sc)[:topk]
            boxes, cls, sc = boxes[o], cls[o], sc[o]
        xyxy = np.stack([boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2,
                         boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2], 1)
        keep = nms(xyxy, sc, cls, iou)
        return np.concatenate([xyxy[keep], sc[keep, None], cls[keep, None]], 1).astype(np.float32)

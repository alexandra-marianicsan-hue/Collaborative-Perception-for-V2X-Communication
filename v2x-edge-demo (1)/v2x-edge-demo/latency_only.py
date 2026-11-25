"""Quick latency test of a .tflite model (no dataset needed).
  python3 latency_only.py models/yolov8n_int8.tflite [/usr/lib/libvx_delegate.so]"""
import resource
import sys
import time

import numpy as np

from tflite_utils import TfliteDetector

d = TfliteDetector(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None, 4)
print("input layout:", d.layout, "| size:", d.imgsz, "| in dtype:", d.in_dtype.__name__ if hasattr(d.in_dtype, "__name__") else d.in_dtype)
x = np.random.randint(0, 255, (d.imgsz, d.imgsz, 3), dtype=np.uint8)
t = time.perf_counter(); d.run(x); print("cold start ms:", round((time.perf_counter() - t) * 1000))
for _ in range(5):
    d.run(x)
ts = [d.run(x)[2] for _ in range(50)]
print(sys.argv[1], "| invoke mean %.1f ms | p95 %.1f ms | %.1f FPS" % (np.mean(ts), np.percentile(ts, 95), 1000 / np.mean(ts)))
print("peak RAM (RSS): %.0f MB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))

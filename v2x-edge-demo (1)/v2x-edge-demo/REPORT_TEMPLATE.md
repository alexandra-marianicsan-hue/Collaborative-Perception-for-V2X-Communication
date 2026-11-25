# Answers to the professor's questions (fill in with your measured numbers)

**Hardware constraints (i.MX 8M Plus, verify on your board):** 4x Cortex-A53 (~1.8 GHz), NPU ~2.3 TOPS (INT8),
LPDDR4 RAM (`free -m`), flash/SD (`df -h`), power (measure with a USB power meter: idle vs inference).
Model footprint: size MB (from `results/benchmark.md`), RAM at runtime (`/usr/bin/time -v` or `top`).

**Tech stack:** Ultralytics YOLOv8n (training, PyTorch) -> TFLite export (FP32/FP16/INT8) -> TFLite runtime
(`tflite_runtime`) on the board; CPU = XNNPACK 4 threads; NPU = VX delegate (`/usr/lib/libvx_delegate.so`).
Input 320x320, single class `vehicle`, OPV2V front-camera images.

**NPU vs CPU:** table `results/benchmark.md` (rows int8-cpu vs int8-npu: total ms, FPS, speed-up = cpu/npu).
Report cold start (graph compile) separately from steady state.

**Optimizations used:** (1) small backbone (YOLOv8n) + transfer learning with frozen backbone; (2) reduced input
resolution (640 -> 320); (3) post-training INT8 quantization (needs calibration images) and FP16;
(4) operator fusion done by the TFLite converter (Conv+BN+activation); (5) late fusion sends detections, not images.
Not used: pruning, knowledge distillation (say so, mention as future work).
**Effect on results:** compare rows fp32 / fp16 / int8 for AP50, F1 (before vs after quantization) and size/latency.

**Bottleneck:** read the columns preproc / invoke / postproc of `results/benchmark.md`. Typical candidates:
CPU post-processing (decode + NMS in numpy), ops not supported by the NPU that fall back to CPU (check the
"partially executed by the delegate" message of benchmark_model), memory bandwidth (LPDDR4), first-inference graph
compilation. State which one dominates in YOUR numbers and what you would change (e.g. move NMS into the model,
smaller resolution, per-channel quantization-aware training).

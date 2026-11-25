# Collaborative Perception for V2X on the Edge (i.MX 8M Plus)

Edge Computing Applications – Project 11 *Collaborative Perception for V2X Communication* (Nicsan & Amanolesei).

Two simulated vehicles each see a different, overlapping part of a road scene. Each runs a **fine-tuned YOLOv8n**
locally on the edge board (NXP i.MX 8M Plus EVK) and only the compact **detections** are shared and fused (late fusion),
instead of raw images (cf. Where2comm, F-Cooper, V2VNet, OPV2V).

> Status (Demo 1, proof of concept): the two views are **simulated** by splitting an image into two overlapping crops.
> Real multi-vehicle OPV2V data is the next step (scripts in the repo, not used for the results below).

## Results
| item | value |
|---|---|
| Model | YOLOv8n, 5 vehicle classes, input 320x320, 3.0 M params, 12.1 MB (`.tflite`, FP32) |
| Fine-tuning | transfer learning from COCO weights, backbone frozen (10 layers), 25 epochs |
| Validation (130 images) | mAP50 **0.431**, mAP50-95 0.281, precision 0.62, recall 0.39 |
| Per class mAP50 | bicycle 0.32, car 0.37, motorcycle 0.46, bus 0.61, truck 0.41 |
| Board inference (CPU, 4 threads, XNNPACK) | ~95 ms per image, ~10.5 FPS |
| Uplink per scene | JPEG ~77 kB vs detections ~31 B (~2500x less) |

Not done / limitations: NPU vs CPU comparison and INT8 quantization (the board's original image has no VX delegate and its
`tflite_runtime` 2.11 cannot run the INT8 models exported with LiteRT 2.3); no pruning/distillation; energy not measured;
small training set; simulated views.

## Dataset
COCO val2017 (public), filtered to images with bicycle/car/motorcycle/bus/truck: 737 train / 130 val images
(`prepare_coco_vehicles.py`, statistics in `data/DATASET_STATS.md`, seed 0).

## Reproduce
**Train + export (Colab/laptop):**
```bash
pip install -r requirements.txt
python prepare_coco_vehicles.py
python train.py --data data/vehicles_big.yaml --epochs 25 --imgsz 320 --batch 32 --freeze 10
python export_tflite.py --weights models/best.pt --imgsz 320 --fp32-only      # -> models/best.tflite
```
**Edge board (i.MX 8M Plus):**
```bash
pip3 install --only-binary=:all: -r requirements-board.txt
python3 latency_only.py models/yolov8n_vehicles_ft.tflite                     # latency, FPS, peak RAM
python3 demo_board.py --model models/yolov8n_vehicles_ft.tflite --images demo_images   # live demo
```
`demo_board.py` writes annotated images to `results/demo/` (top: vehicle A orange + vehicle B blue, bottom: fused green)
and prints latency and uplink bytes. Connection tips for the board: `BOARD_SETUP.md`.

## Files
| group | files |
|---|---|
| Demo on the board | `demo_board.py`, `tflite_utils.py`, `collab.py`, `bench_tflite.py`, `latency_only.py` |
| Data + training (laptop/Colab) | `prepare_coco_vehicles.py`, `train.py`, `export_tflite.py`, `make_pack_coco.py` |
| OPV2V (next step, not used in results) | `prepare_opv2v.py`, `verify_labels.py`, `opv2v_geometry.py`, `collab_opv2v.py`, `make_pack.py` |
| First version with ONNX (legacy) | `detector.py`, `demo.py`, `evaluate.py`, `benchmark.py`, `export.py`, `prepare_data.py` |
| Report | `REPORT_TEMPLATE.md` |

## Part 2 – cross-view association on nuScenes

Neighbouring cameras of one car share ~15-20 degrees of view, so a vehicle at the seam is cut in half: one part in each image.
`assoc.py` decides that the two parts are the **same vehicle** by combining three cues – geometry in a common ground-plane
frame (shared calibration + pose), HSV colour appearance and ORB keypoint matches – and merges them into one object.
Full guide (steps, results tables, edge stack, limitations): **[PRACTICAL2.md](PRACTICAL2.md)**.

| file | purpose |
|---|---|
| `assoc.py` | association (geo / app / kp cues, greedy one-to-one matching, object merging) |
| `nusc_geometry.py` | nuScenes-like camera rig, 3D box -> 2D projection |
| `prepare_nuscenes.py` | nuScenes -> evaluation pack (pairs + ground-truth identities) and a YOLO fine-tuning set |
| `eval_assoc.py` | precision/recall/F1, recall on cut objects, count error, ms/pair; `--tune`, `--pose-noise-m/-deg` |
| `demo_assoc.py` | live demo on the board (coloured links, BEV map, per-stage latency); `--pair A B` for your own photos |
| `detect_utils.py` | TFLite detector wrapper with per-stage timing |
| `test_assoc_synth.py`, `make_fake_nuscenes.py` | code-path tests without the dataset (synthetic world / fake nuScenes) |
| `v2x_nuscenes_finetune.ipynb` (Colab) | fine-tuning on nuScenes, int8 export, before/after tables |

Quick checks without any dataset: `python test_assoc_synth.py --kp` and the dry-run in PRACTICAL2.md section 3B.

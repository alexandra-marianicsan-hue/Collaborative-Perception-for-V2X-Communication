"""Collaborative-perception utilities: views, box fusion, bandwidth, matching.

Detections are numpy arrays of shape (N, 6): x1, y1, x2, y2, conf, cls (pixels).
"""
import numpy as np


def iou_matrix(a, b, mode="iou"):
    """Pairwise overlap. mode='iou' or 'iom' (intersection over the smaller box)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    if mode == "iom":
        denom = np.minimum(area_a[:, None], area_b[None, :])
    else:
        denom = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(denom, 1e-9)


def nms(boxes, scores, classes, thr=0.5, mode="iou"):
    """Class-aware greedy NMS. Returns indices to keep."""
    keep = []
    for c in np.unique(classes):
        idx = np.where(classes == c)[0]
        order = idx[np.argsort(-scores[idx])]
        while len(order):
            i = order[0]
            keep.append(i)
            if len(order) == 1:
                break
            ov = iou_matrix(boxes[i:i + 1], boxes[order[1:]], mode)[0]
            order = order[1:][ov < thr]
    return np.array(keep, dtype=int)


def make_views(frame, overlap=0.2):
    """Simulate two vehicles seeing different, overlapping parts of the scene.

    Vehicle A sees the left (0.5+overlap/2) of the image, vehicle B the right part.
    Returns [(view_img, (x_offset, y_offset)), ...].
    """
    W = frame.shape[1]
    a_end = int(W * (0.5 + overlap / 2))
    b_start = int(W * (0.5 - overlap / 2))
    return [(frame[:, :a_end], (0, 0)), (frame[:, b_start:], (b_start, 0))]


def shift(dets, offset):
    """Move detections from a view's coordinates to the global frame."""
    d = dets.copy()
    d[:, [0, 2]] += offset[0]
    d[:, [1, 3]] += offset[1]
    return d


def fuse(det_list, thr=0.8):
    """Late fusion: concatenate global detections, drop cross-view duplicates.

    Uses intersection-over-min so a box truncated at a view border is merged
    with the full box seen by the other vehicle.
    """
    det_list = [d for d in det_list if len(d)]
    if not det_list:
        return np.zeros((0, 6), np.float32)
    d = np.concatenate(det_list, 0)
    keep = nms(d[:, :4], d[:, 4], d[:, 5], thr, mode="iom")
    return d[keep]


# ---- bandwidth -------------------------------------------------------------
def jpeg_bytes(img, quality=80):
    import cv2  # lazy: not needed on the board
    return len(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1])


def det_msg_bytes(dets):
    """Compact message: 4-byte header + per box 4x int16 coords + uint8 conf + uint8 cls."""
    return 4 + 10 * len(dets)


# ---- ground truth / matching ----------------------------------------------
def load_gt(label_path, W, H):
    """YOLO label file -> array (M, 5): cls, x1, y1, x2, y2 in pixels."""
    rows = []
    try:
        for line in open(label_path):
            p = line.split()
            if len(p) < 5:
                continue
            c, cx, cy, w, h = int(float(p[0])), *map(float, p[1:5])
            rows.append([c, (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
    except FileNotFoundError:
        pass
    return np.array(rows, dtype=np.float64).reshape(-1, 5)


def match_counts(dets, gt, iou_thr=0.5):
    """Greedy class-aware matching. Returns (tp, fp, fn)."""
    used = set()
    tp = 0
    for i in np.argsort(-dets[:, 4]) if len(dets) else []:
        if not len(gt):
            break
        ious = iou_matrix(dets[i:i + 1, :4], gt[:, 1:5])[0]
        best, bj = 0.0, -1
        for j in range(len(gt)):
            if j in used or int(gt[j, 0]) != int(dets[i, 5]):
                continue
            if ious[j] > best:
                best, bj = ious[j], j
        if bj >= 0 and best >= iou_thr:
            used.add(bj)
            tp += 1
    return tp, len(dets) - tp, len(gt) - tp

"""3D box -> 2D box projection and nuScenes-like camera rig (numpy only).

Used by prepare_nuscenes.py (ground truth) and by the synthetic test. Frames: ego x forward, y left,
z up; camera x right, y down, z forward.
"""
import numpy as np

# nuScenes camera rig (approximate, ego frame): name -> (x, y, z, yaw_deg, fx)  [image 1600x900]
RIG = {
    "CAM_FRONT":       (1.70,  0.00, 1.51,    0.0, 1266.4),
    "CAM_FRONT_LEFT":  (1.52,  0.50, 1.51,   55.0, 1272.6),
    "CAM_FRONT_RIGHT": (1.58, -0.50, 1.51,  -55.0, 1260.8),
    "CAM_BACK":        (0.00,  0.00, 1.55,  180.0,  809.7),
    "CAM_BACK_LEFT":   (1.04,  0.48, 1.59,  110.0, 1256.7),
    "CAM_BACK_RIGHT":  (1.05, -0.46, 1.59, -110.0, 1259.0),
}
# adjacent pairs around the car (neighbouring cameras share ~15-20 degrees of view)
PAIRS = [("CAM_FRONT_LEFT", "CAM_FRONT"), ("CAM_FRONT", "CAM_FRONT_RIGHT"),
         ("CAM_FRONT_RIGHT", "CAM_BACK_RIGHT"), ("CAM_BACK_RIGHT", "CAM_BACK"),
         ("CAM_BACK", "CAM_BACK_LEFT"), ("CAM_BACK_LEFT", "CAM_FRONT_LEFT")]


def cam_to_ego(x, y, z, yaw_deg):
    """Camera looking horizontally along yaw (camera x right, y down, z forward) -> 4x4 camera->ego."""
    p = np.radians(yaw_deg)
    fwd = np.array([np.cos(p), np.sin(p), 0.0])
    right = np.array([np.sin(p), -np.cos(p), 0.0])
    down = np.array([0.0, 0.0, -1.0])
    T = np.eye(4)
    T[:3, :3] = np.stack([right, down, fwd], 1)
    T[:3, 3] = [x, y, z]
    return T


def intrinsics(fx, W=1600, H=900):
    return np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1.0]])


def box_corners(center, wlh, yaw):
    """8 corners (3x8) of a nuScenes box: length along heading, width across, z up."""
    w, l, h = wlh
    x = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1.0])
    y = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1.0])
    z = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1.0])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    return R @ np.stack([x, y, z]) + np.asarray(center, dtype=np.float64).reshape(3, 1)


_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def project_box(corners_cam, K, W, H, near=0.3):
    """Project a 3D box (3x8, camera frame) to the image.

    Edges are clipped against the near plane, so boxes that straddle the camera plane still work.
    Returns None if nothing is in front of the camera, else
    dict(box=[x1,y1,x2,y2] clipped to the image, full_area, clip_area, depth).
    """
    pts = []
    for i, j in _EDGES:
        a, b = corners_cam[:, i], corners_cam[:, j]
        if a[2] >= near and b[2] >= near:
            pts += [a, b]
        elif a[2] >= near or b[2] >= near:
            if a[2] < near:
                a, b = b, a                                    # a is in front
            t = (near - a[2]) / (b[2] - a[2])
            pts += [a, a + t * (b - a)]
    if not pts:
        return None
    P = np.array(pts).T
    uv = (K @ P)[:2] / P[2]
    x1, y1, x2, y2 = uv[0].min(), uv[1].min(), uv[0].max(), uv[1].max()
    full = max(x2 - x1, 0) * max(y2 - y1, 0)
    cx1, cy1, cx2, cy2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    clip = (cx2 - cx1) * (cy2 - cy1)
    return {"box": [cx1, cy1, cx2, cy2], "full_area": full, "clip_area": clip,
            "depth": float(corners_cam[2].mean())}

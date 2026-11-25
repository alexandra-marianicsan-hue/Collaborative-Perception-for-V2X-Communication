"""CARLA / OPV2V geometry: 3D vehicle boxes -> 2D camera boxes, and 2D -> ground-plane BEV.

OPV2V yaml conventions (see OpenCOOD data annotation tutorial):
  camera0.cords     = [x, y, z, roll, yaw, pitch] of the camera in the CARLA world (degrees)
  camera0.intrinsic = 3x3 K
  vehicles[id]      = {location, center, angle=[roll,yaw,pitch], extent=half sizes (l,w,h)}
CARLA camera frame: x forward, y right, z up  ->  OpenCV frame: (y, -z, x).
"""
import numpy as np


def carla_rotation(roll, yaw, pitch):
    cr, sr = np.cos(np.radians(roll)), np.sin(np.radians(roll))
    cy, sy = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    cp, sp = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
    return np.array([[cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
                     [sp, -cp * sr, cp * cr]])


def vehicle_center_world(veh):
    R = carla_rotation(*veh["angle"])
    return np.array(veh["location"], float) + R @ np.array(veh["center"], float)


def vehicle_corners_world(veh):
    R = carla_rotation(*veh["angle"])
    c = vehicle_center_world(veh)
    e = np.array(veh["extent"], float)
    signs = np.array([[a, b, d] for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)], float)
    return c + (signs * e) @ R.T


def world_to_cam(pts, pose):
    """World points -> OpenCV camera frame (x right, y down, z forward)."""
    R = carla_rotation(*pose[3:6])
    pc = (np.asarray(pts, float) - np.array(pose[:3], float)) @ R      # R^T (p - t)
    return np.stack([pc[:, 1], -pc[:, 2], pc[:, 0]], 1)


def project(pts, pose, K):
    pcv = world_to_cam(pts, pose)
    z = pcv[:, 2]
    uvw = (np.asarray(K, float) @ pcv.T).T
    uv = uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-6)
    return uv, z


def box2d(veh, pose, K, W, H, min_visible=0.5, min_size=10):
    """2D box (x1,y1,x2,y2 px) of a 3D vehicle box, or None if behind/too small/mostly outside."""
    uv, z = project(vehicle_corners_world(veh), pose, K)
    if (z <= 0.5).any():
        return None
    x1, y1 = uv.min(0)
    x2, y2 = uv.max(0)
    full = max(x2 - x1, 1e-6) * max(y2 - y1, 1e-6)
    cx1, cy1, cx2, cy2 = max(x1, 0), max(y1, 0), min(x2, W - 1), min(y2, H - 1)
    if cx2 - cx1 < min_size or cy2 - cy1 < min_size:
        return None
    if (cx2 - cx1) * (cy2 - cy1) / full < min_visible:
        return None
    return [float(cx1), float(cy1), float(cx2), float(cy2)]


def backproject_ground(u, v, pose, K, ground_z):
    """Pixel (u,v) -> world point on the plane z=ground_z (or None)."""
    d_cv = np.linalg.inv(np.asarray(K, float)) @ np.array([u, v, 1.0])
    d_carla = np.array([d_cv[2], d_cv[0], -d_cv[1]])
    d_w = carla_rotation(*pose[3:6]) @ d_carla
    if d_w[2] >= -1e-6:
        return None
    t = (ground_z - pose[2]) / d_w[2]
    if t <= 0:
        return None
    return np.array(pose[:3], float) + t * d_w

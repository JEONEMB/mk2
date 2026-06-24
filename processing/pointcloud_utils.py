"""Metric point-cloud helpers. Input depth is always in millimetres."""

from __future__ import annotations

import numpy as np

from camera.frame_types import CameraIntrinsics


def depth_to_pointcloud(depth_mm: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Back-project a cleaned depth image into an ``H x W x 3`` millimetre grid."""

    depth = np.asarray(depth_mm, dtype=np.float32)
    intrinsics.validate_for(depth.shape)
    yy, xx = np.indices(depth.shape, dtype=np.float32)
    points = np.empty((*depth.shape, 3), dtype=np.float32)
    points[..., 2] = depth
    points[..., 0] = (xx - intrinsics.cx) * depth / intrinsics.fx
    points[..., 1] = (yy - intrinsics.cy) * depth / intrinsics.fy
    points[~np.isfinite(depth)] = np.nan
    return points


def rescale_pointcloud_to_depth(native_points_xyz: np.ndarray, depth_mm: np.ndarray) -> np.ndarray:
    """Keep SDK point-cloud rays while replacing Z with filtered/scaled depth.

    Synexens' native point cloud already incorporates its camera model.  Scaling
    points along each native ray preserves that geometry and makes the result
    exactly consistent with the depth image used by the measurement pipeline.
    """

    native = np.asarray(native_points_xyz, dtype=np.float32)
    depth = np.asarray(depth_mm, dtype=np.float32)
    if native.shape != (*depth.shape, 3):
        raise ValueError("native point cloud must match the depth image shape")
    native_z = native[..., 2]
    valid = np.isfinite(native).all(axis=2) & np.isfinite(depth) & (native_z > 0)
    points = np.full_like(native, np.nan, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        rays = native / native_z[..., None]
        points[valid] = rays[valid] * depth[valid, None]
    return points


def valid_points(points_xyz: np.ndarray, min_z_mm: float = 0.0, max_z_mm: float = np.inf) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] >= min_z_mm) & (points[:, 2] <= max_z_mm)
    return points[valid]


def reshape_pointcloud(points_xyz: np.ndarray, frame_height: int, frame_width: int) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float32)
    if points.size != frame_height * frame_width * 3:
        raise ValueError("point cloud size does not match frame dimensions")
    return points.reshape(frame_height, frame_width, 3)


def extract_roi_points(points_grid: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = roi
    grid = np.asarray(points_grid)
    if grid.ndim != 3 or grid.shape[2] != 3:
        raise ValueError("points_grid must have shape H x W x 3")
    return valid_points(grid[y : y + height, x : x + width])


def sample_points(points: np.ndarray, max_points: int, rng: np.random.Generator | None = None) -> np.ndarray:
    points = np.asarray(points)
    if len(points) <= max_points:
        return points
    generator = rng or np.random.default_rng(0)
    return points[generator.choice(len(points), max_points, replace=False)]


def compute_plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return orthonormal in-plane u/v axes for a unit or non-unit normal."""

    n = np.asarray(normal, dtype=np.float64)
    n /= np.linalg.norm(n)
    seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    axis_u = np.cross(n, seed)
    axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(n, axis_u)
    return axis_u.astype(np.float64), axis_v.astype(np.float64)


def project_to_plane_basis(
    points: np.ndarray, origin: np.ndarray, axis_u: np.ndarray, axis_v: np.ndarray
) -> np.ndarray:
    delta = np.asarray(points, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    return np.column_stack((delta @ axis_u, delta @ axis_v))

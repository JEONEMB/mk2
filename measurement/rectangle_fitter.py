"""Fit an oriented rectangle to validated top-surface points."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from calibration.platform_calibrator import PlatformModel
from processing.pointcloud_utils import compute_plane_basis, project_to_plane_basis


@dataclass(frozen=True)
class RotatedRectangle:
    center_uv: tuple[float, float]
    size_mm: tuple[float, float]
    angle_deg: float
    corners_uv: np.ndarray


def project_top_points_to_uv(top_points: np.ndarray, platform_model: PlatformModel) -> np.ndarray:
    normal = np.asarray(platform_model.plane_normal, dtype=np.float64)
    origin = -platform_model.plane_d * normal  # closest point on n·x+d=0
    axis_u, axis_v = compute_plane_basis(normal)
    return project_to_plane_basis(top_points, origin, axis_u, axis_v)


def trim_uv_outliers(uv_points: np.ndarray, low_percentile: float, high_percentile: float) -> np.ndarray:
    uv = np.asarray(uv_points, dtype=np.float64)
    if len(uv) < 4:
        return uv
    low = np.percentile(uv, low_percentile, axis=0)
    high = np.percentile(uv, high_percentile, axis=0)
    trimmed = uv[np.all((uv >= low) & (uv <= high), axis=1)]
    return trimmed if len(trimmed) >= 4 else uv


def fit_rotated_rectangle(uv_points: np.ndarray) -> RotatedRectangle:
    uv = np.asarray(uv_points, dtype=np.float32)
    if len(uv) < 4:
        raise ValueError("at least four top points are required for a rectangle")
    (cx, cy), (width, height), angle = cv2.minAreaRect(uv.reshape(-1, 1, 2))
    corners = cv2.boxPoints(((cx, cy), (width, height), angle)).astype(np.float64)
    return RotatedRectangle((float(cx), float(cy)), (float(width), float(height)), float(angle), corners)


def enforce_rectangular_prior(rect: RotatedRectangle) -> RotatedRectangle:
    """Retain the minimum-area rectangle; its four right angles are the prior."""

    if min(rect.size_mm) <= 0:
        raise ValueError("fitted rectangle has zero area")
    return rect


def compute_width_length(rect: RotatedRectangle) -> tuple[float, float]:
    first, second = rect.size_mm
    return (min(first, second), max(first, second))


class RectangleFitter:
    def __init__(self, measurement_config: object) -> None:
        self.config = measurement_config

    def fit(self, top_points: np.ndarray, platform_model: PlatformModel) -> RotatedRectangle:
        uv = project_top_points_to_uv(top_points, platform_model)
        uv = trim_uv_outliers(
            uv, self.config.rectangle_trim_percentile_low, self.config.rectangle_trim_percentile_high
        )
        return enforce_rectangular_prior(fit_rotated_rectangle(uv))

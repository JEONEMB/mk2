"""Extract and validate the box's upper planar surface."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from calibration.platform_calibrator import PlatformModel
from processing.plane_fitting import PlaneFit, fit_plane_ransac, point_to_plane_signed_distance


@dataclass(frozen=True)
class TopSurfaceResult:
    success: bool
    top_height_mm: float | None
    points: np.ndarray
    plane: PlaneFit | None
    residual_std_mm: float | None
    inlier_ratio: float
    warning: str | None = None


def estimate_top_height_mode(box_heights: np.ndarray) -> float:
    values = np.asarray(box_heights, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("box has no finite height values")
    # 1 mm bins retain a physically meaningful, stable height mode.
    lower, upper = np.floor(values.min()), np.ceil(values.max()) + 1
    if upper - lower <= 1:
        return float(np.median(values))
    counts, edges = np.histogram(values, bins=np.arange(lower, upper + 1.0, 1.0))
    index = int(np.argmax(counts))
    return float((edges[index] + edges[index + 1]) / 2)


def select_top_band_points(
    box_points: np.ndarray, box_heights: np.ndarray, top_height: float, band_mm: float
) -> tuple[np.ndarray, np.ndarray]:
    heights = np.asarray(box_heights)
    mask = np.isfinite(heights) & (np.abs(heights - top_height) <= band_mm)
    return np.asarray(box_points)[mask], mask


def fit_top_plane(top_points: np.ndarray, platform_model: PlatformModel, config: object) -> PlaneFit:
    del platform_model  # Kept in the signature because the surface is platform-relative.
    return fit_plane_ransac(
        top_points,
        iterations=200,
        threshold_mm=config.top_plane_threshold_mm,
        min_inliers=config.min_top_points,
    )


def validate_top_plane(residuals: np.ndarray, inlier_ratio: float, config: object) -> tuple[bool, float, str | None]:
    residual_std = float(np.std(residuals))
    if residual_std > config.max_top_plane_residual_std_mm:
        return False, residual_std, f"top-plane residual is {residual_std:.2f} mm"
    if inlier_ratio < config.min_top_inlier_ratio:
        return False, residual_std, f"top-plane inlier ratio is {inlier_ratio:.2f}"
    return True, residual_std, None


class TopSurfaceExtractor:
    def __init__(self, measurement_config: object) -> None:
        self.config = measurement_config

    def extract(self, box_points: np.ndarray, box_heights: np.ndarray, platform_model: PlatformModel) -> TopSurfaceResult:
        if len(box_points) < self.config.min_top_points:
            return TopSurfaceResult(False, None, np.empty((0, 3)), None, None, 0.0, "too few box points")
        try:
            mode_height = estimate_top_height_mode(box_heights)
            band_points, _ = select_top_band_points(
                box_points, box_heights, mode_height, self.config.top_band_mm
            )
            if len(band_points) < self.config.min_top_points:
                return TopSurfaceResult(False, None, band_points, None, None, 0.0, "too few top-band points")
            plane = fit_top_plane(band_points, platform_model, self.config)
            inlier_points = band_points[plane.inlier_mask]
            residuals = point_to_plane_signed_distance(inlier_points, plane.normal, plane.d)
            ratio = len(inlier_points) / len(band_points)
            valid, residual_std, warning = validate_top_plane(residuals, ratio, self.config)
            if not valid:
                return TopSurfaceResult(False, None, inlier_points, plane, residual_std, ratio, warning)
            # Use all validated top-plane inliers to estimate the platform-relative height.
            height = float(np.median(point_to_plane_signed_distance(
                inlier_points, platform_model.plane_normal, platform_model.plane_d
            )))
            return TopSurfaceResult(True, height, inlier_points, plane, residual_std, ratio)
        except ValueError as exc:
            return TopSurfaceResult(False, None, np.empty((0, 3)), None, None, 0.0, str(exc))

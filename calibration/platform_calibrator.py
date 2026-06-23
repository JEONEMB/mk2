"""Build and maintain a metric model of the empty platform."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

import numpy as np

from processing.plane_fitting import fit_plane_ransac, point_to_plane_signed_distance, update_plane_d_only
from processing.pointcloud_utils import extract_roi_points, sample_points, valid_points


@dataclass(frozen=True)
class PlatformModel:
    frame_size: tuple[int, int]  # width, height
    resolution: str
    plane_normal: np.ndarray
    plane_d: float
    camera_height_mm: float
    measurement_roi: tuple[int, int, int, int]
    residual_std_mm: float
    created_at: str
    version: str = "mk2"

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "frame_size": list(self.frame_size),
            "resolution": self.resolution,
            "plane_normal": self.plane_normal.tolist(),
            "plane_d": self.plane_d,
            "camera_height_mm": self.camera_height_mm,
            "measurement_roi": list(self.measurement_roi),
            "residual_std_mm": self.residual_std_mm,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "PlatformModel":
        return cls(
            frame_size=tuple(int(x) for x in data["frame_size"]),  # type: ignore[arg-type]
            resolution=str(data.get("resolution", "")),
            plane_normal=np.asarray(data["plane_normal"], dtype=np.float64),
            plane_d=float(data["plane_d"]),
            camera_height_mm=float(data["camera_height_mm"]),
            measurement_roi=tuple(int(x) for x in data["measurement_roi"]),  # type: ignore[arg-type]
            residual_std_mm=float(data.get("residual_std_mm", 0.0)),
            created_at=str(data.get("created_at", "")),
            version=str(data.get("version", "mk2")),
        )


class PlatformCalibrator:
    """Fits the platform from cleaned, calibrated point-cloud grids only."""

    def __init__(self, calibration_config: object) -> None:
        self.config = calibration_config

    def collect_empty_platform_frame(self, points_grid: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
        return extract_roi_points(points_grid, roi)

    def calibrate(
        self,
        pointcloud_frames: Sequence[np.ndarray],
        *,
        resolution: str = "",
        manual_roi: tuple[int, int, int, int] | None = None,
    ) -> PlatformModel:
        return self.build_platform_model(pointcloud_frames, resolution=resolution, manual_roi=manual_roi)

    def build_platform_model(
        self,
        pointcloud_frames: Sequence[np.ndarray],
        *,
        resolution: str = "",
        manual_roi: tuple[int, int, int, int] | None = None,
    ) -> PlatformModel:
        if not pointcloud_frames:
            raise ValueError("empty-platform calibration requires at least one point-cloud frame")
        first = np.asarray(pointcloud_frames[0])
        if first.ndim != 3 or first.shape[2] != 3:
            raise ValueError("calibration frames must be H x W x 3 point-cloud grids")
        height, width = first.shape[:2]
        roi = manual_roi or self._central_roi(width, height)
        points: list[np.ndarray] = []
        for grid in pointcloud_frames:
            array = np.asarray(grid)
            if array.shape != first.shape:
                raise ValueError("all calibration point clouds must share one resolution")
            points.append(self.collect_empty_platform_frame(array, roi))
        merged = sample_points(np.concatenate(points), self.config.max_sample_points)
        fit = fit_plane_ransac(
            merged,
            iterations=self.config.ransac_iterations,
            threshold_mm=self.config.ransac_threshold_mm,
            min_inliers=self.config.min_inliers,
        )
        # n is unit length, so |d| is the camera-origin distance to the plane.
        return PlatformModel(
            frame_size=(width, height),
            resolution=resolution or f"{width}x{height}",
            plane_normal=fit.normal,
            plane_d=fit.d,
            camera_height_mm=abs(fit.d),
            measurement_roi=roi,
            residual_std_mm=fit.residual_std_mm,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def update_floor_for_measurement(
        self,
        current_points: np.ndarray,
        base_platform_model: PlatformModel,
        box_mask: np.ndarray | None = None,
    ) -> PlatformModel:
        """Optionally correct a small camera/platform offset without rotating the plane."""

        grid = np.asarray(current_points)
        x, y, width, height = base_platform_model.measurement_roi
        roi_grid = grid[y : y + height, x : x + width]
        if box_mask is not None:
            if box_mask.shape != grid.shape[:2]:
                raise ValueError("box_mask must match the point-cloud image size")
            roi_grid = roi_grid.copy()
            roi_grid[box_mask[y : y + height, x : x + width]] = np.nan
        points = valid_points(roi_grid)
        new_d = update_plane_d_only(
            points,
            base_platform_model.plane_normal,
            base_platform_model.plane_d,
            self.config.dynamic_floor_update_threshold_mm,
        )
        if abs(new_d - base_platform_model.plane_d) > self.config.dynamic_floor_update_threshold_mm:
            new_d = base_platform_model.plane_d
        residuals = point_to_plane_signed_distance(points, base_platform_model.plane_normal, new_d)
        return PlatformModel(
            frame_size=base_platform_model.frame_size,
            resolution=base_platform_model.resolution,
            plane_normal=base_platform_model.plane_normal,
            plane_d=new_d,
            camera_height_mm=abs(new_d),
            measurement_roi=base_platform_model.measurement_roi,
            residual_std_mm=float(np.std(residuals)) if len(residuals) else float("nan"),
            created_at=base_platform_model.created_at,
            version=base_platform_model.version,
        )

    def _central_roi(self, width: int, height: int) -> tuple[int, int, int, int]:
        ratio = self.config.roi_ratio
        roi_width, roi_height = int(width * ratio), int(height * ratio)
        return ((width - roi_width) // 2, (height - roi_height) // 2, roi_width, roi_height)

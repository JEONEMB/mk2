"""Build and maintain a metric model of the empty platform."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Sequence

import cv2
import numpy as np

from processing.plane_fitting import PlaneFit, fit_plane_ransac, fit_plane_svd, point_to_plane_signed_distance
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
    platform_mask: np.ndarray | None = field(default=None, repr=False, compare=False)
    normal_alignment: float = 0.0
    roi_area_ratio: float = 0.0
    # The SDK-only plane estimate is retained for diagnostics.  When a ruler
    # reference is supplied, ``camera_height_mm`` is the scaled, usable value.
    native_camera_height_mm: float | None = None
    applied_depth_scale: float = 1.0
    residual_p05_mm: float = 0.0
    residual_p50_mm: float = 0.0
    residual_p95_mm: float = 0.0

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
            "normal_alignment": self.normal_alignment,
            "roi_area_ratio": self.roi_area_ratio,
            "native_camera_height_mm": self.native_camera_height_mm,
            "applied_depth_scale": self.applied_depth_scale,
            "residual_p05_mm": self.residual_p05_mm,
            "residual_p50_mm": self.residual_p50_mm,
            "residual_p95_mm": self.residual_p95_mm,
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
            normal_alignment=float(data.get("normal_alignment", 0.0)),
            roi_area_ratio=float(data.get("roi_area_ratio", 0.0)),
            native_camera_height_mm=(
                float(data["native_camera_height_mm"])
                if data.get("native_camera_height_mm") is not None
                else None
            ),
            applied_depth_scale=float(data.get("applied_depth_scale", 1.0)),
            residual_p05_mm=float(data.get("residual_p05_mm", 0.0)),
            residual_p50_mm=float(data.get("residual_p50_mm", 0.0)),
            residual_p95_mm=float(data.get("residual_p95_mm", 0.0)),
        )


class PlatformCalibrator:
    """Fits the platform from cleaned, calibrated point-cloud grids only."""

    def __init__(self, calibration_config: object) -> None:
        self.config = calibration_config

    def collect_empty_platform_frame(
        self, points_grid: np.ndarray, roi: tuple[int, int, int, int] | None = None
    ) -> np.ndarray:
        """Keep only farthest metric candidates before fitting the platform plane.

        A box or loose obstacle is normally closer to the overhead camera than
        the empty platform. Walls may remain in this first selection, but the
        later horizontal-normal RANSAC condition excludes them.
        """

        if roi is not None:
            return extract_roi_points(points_grid, roi)
        grid = np.asarray(points_grid, dtype=np.float64)
        valid = np.isfinite(grid).all(axis=2)
        distances = np.linalg.norm(grid, axis=2)
        finite_distances = distances[valid]
        if not len(finite_distances):
            return np.empty((0, 3), dtype=np.float64)
        cutoff = np.percentile(finite_distances, self.config.platform_farthest_percentile)
        return grid[valid & (distances >= cutoff)]

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
        """Build a platform from a coarse seed, then refit it over full-frame inliers."""

        if not pointcloud_frames:
            raise ValueError("empty-platform calibration requires at least one point-cloud frame")
        first = np.asarray(pointcloud_frames[0])
        if first.ndim != 3 or first.shape[2] != 3:
            raise ValueError("calibration frames must be H x W x 3 point-cloud grids")
        height, width = first.shape[:2]
        points: list[np.ndarray] = []
        valid_ratios: list[float] = []
        for grid in pointcloud_frames:
            array = np.asarray(grid)
            if array.shape != first.shape:
                raise ValueError("all calibration point clouds must share one resolution")
            valid_ratios.append(float(np.isfinite(array).all(axis=2).mean()))
            points.append(self.collect_empty_platform_frame(array))
        mean_valid_ratio = float(np.mean(valid_ratios))
        if mean_valid_ratio < self.config.min_calibration_valid_depth_ratio:
            raise ValueError(
                f"valid depth is only {mean_valid_ratio:.1%}; adjust camera/IR exposure and aim downward at the platform"
            )
        if not any(len(candidate) for candidate in points):
            raise ValueError("no far-depth platform candidates were found")
        merged = sample_points(np.concatenate(points), self.config.max_sample_points)
        minimum_alignment = float(np.cos(np.deg2rad(self.config.platform_max_tilt_deg)))
        seed_fit = fit_plane_ransac(
            merged,
            iterations=self.config.ransac_iterations,
            threshold_mm=self.config.ransac_threshold_mm,
            min_inliers=self.config.min_inliers,
            expected_normal=np.array([0.0, 0.0, -1.0]),
            min_normal_alignment=minimum_alignment,
        )
        if manual_roi is None:
            # The far-point RANSAC is only a seed.  Refit normal and offset
            # using the stable full-frame platform support before deriving ROI.
            fit, platform_mask = self._refine_full_platform_plane(pointcloud_frames, seed_fit)
            roi = self._inset_measurement_roi(self._largest_safe_rectangle(platform_mask))
        else:
            fit = seed_fit
            roi = manual_roi
            platform_mask = np.zeros((height, width), dtype=bool)
            x, y, roi_width, roi_height = roi
            platform_mask[y : y + roi_height, x : x + roi_width] = True
        roi_area_ratio = (roi[2] * roi[3]) / float(width * height)
        if roi_area_ratio < self.config.min_platform_roi_area_ratio:
            raise ValueError(
                f"platform ROI covers only {roi_area_ratio:.1%} of the frame; "
                "aim the camera downward at an empty platform and calibrate again"
            )
        normal_alignment = abs(float(np.dot(fit.normal, np.array([0.0, 0.0, -1.0]))))
        if normal_alignment < minimum_alignment:
            raise ValueError("refined platform normal is not compatible with an overhead installation")
        residual_p05, residual_p50, residual_p95, residual_std = self._roi_residual_statistics(
            pointcloud_frames, roi, fit.normal, fit.d
        )
        if (
            residual_std > self.config.max_calibration_residual_std_mm
            or max(abs(residual_p05), abs(residual_p95)) > self.config.max_calibration_residual_abs_p95_mm
        ):
            raise ValueError(
                "platform is not planar enough for measurement: "
                f"p05={residual_p05:.1f} mm, p50={residual_p50:.1f} mm, "
                f"p95={residual_p95:.1f} mm, std={residual_std:.1f} mm"
            )
        # n is unit length, so |d| is the camera-origin distance to the plane.
        return PlatformModel(
            frame_size=(width, height),
            resolution=resolution or f"{width}x{height}",
            plane_normal=fit.normal,
            plane_d=fit.d,
            camera_height_mm=abs(fit.d),
            measurement_roi=roi,
            residual_std_mm=residual_std,
            created_at=datetime.now(timezone.utc).isoformat(),
            platform_mask=platform_mask,
            normal_alignment=normal_alignment,
            roi_area_ratio=roi_area_ratio,
            native_camera_height_mm=abs(fit.d),
            residual_p05_mm=residual_p05,
            residual_p50_mm=residual_p50,
            residual_p95_mm=residual_p95,
        )

    def _inset_measurement_roi(self, roi: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """Reserve a guard band so boundary clutter cannot join a box candidate."""

        x, y, width, height = roi
        inset = int(self.config.measurement_roi_inset_px)
        if inset < 0:
            raise ValueError("measurement ROI inset must not be negative")
        if width <= inset * 2 or height <= inset * 2:
            raise ValueError(
                f"platform ROI {width}x{height} is too small for a {inset}px safety margin"
            )
        safe_width, safe_height = width - inset * 2, height - inset * 2
        minimum_side = int(self.config.min_measurement_roi_side_px)
        if safe_width < minimum_side or safe_height < minimum_side:
            raise ValueError(
                f"safe platform ROI is only {safe_width}x{safe_height} pixels; expected at least "
                f"{minimum_side}x{minimum_side}"
            )
        return x + inset, y + inset, safe_width, safe_height

    def _refine_full_platform_plane(
        self, pointcloud_frames: Sequence[np.ndarray], seed_fit: PlaneFit
    ) -> tuple[PlaneFit, np.ndarray]:
        """Expand a far-point seed to stable full-frame support, then refit it.

        The first RANSAC deliberately looks only at distant points to avoid a
        box.  It is not accurate enough to be the stored model: once a
        candidate plane is available, every frame contributes its full-image
        inliers and SVD is run again on that larger, stable support.
        """

        fit = seed_fit
        minimum_alignment = float(np.cos(np.deg2rad(self.config.platform_max_tilt_deg)))
        platform_mask: np.ndarray | None = None
        # Two passes let the first full-frame SVD remove the small bias left by
        # the far-only seed; the second pass rebuilds its support from that
        # refined normal and offset.
        for _ in range(2):
            stable_mask = self._stable_plane_mask(pointcloud_frames, fit.normal, fit.d)
            platform_mask = self._largest_plane_component(stable_mask)
            support = self._collect_masked_plane_inliers(
                pointcloud_frames, platform_mask, fit.normal, fit.d
            )
            if len(support) < self.config.min_inliers:
                raise ValueError(
                    f"full-frame platform support has only {len(support)} points; "
                    f"expected at least {self.config.min_inliers}"
                )
            fit = fit_plane_svd(sample_points(support, self.config.max_sample_points))
            alignment = abs(float(np.dot(fit.normal, np.array([0.0, 0.0, -1.0]))))
            if alignment < minimum_alignment:
                raise ValueError("full-frame platform fit is not horizontal enough for overhead measurement")

        assert platform_mask is not None
        # One final SVD is deliberately performed after rebuilding the support
        # from the refined plane.  The stored normal/d therefore comes from
        # the full-frame component, never from the initial far-point seed.
        stable_mask = self._stable_plane_mask(pointcloud_frames, fit.normal, fit.d)
        platform_mask = self._largest_plane_component(stable_mask)
        final_support = self._collect_masked_plane_inliers(
            pointcloud_frames, platform_mask, fit.normal, fit.d
        )
        if len(final_support) < self.config.min_inliers:
            raise ValueError(
                f"final platform support has only {len(final_support)} points; "
                f"expected at least {self.config.min_inliers}"
            )
        fit = fit_plane_svd(sample_points(final_support, self.config.max_sample_points))
        stable_mask = self._stable_plane_mask(pointcloud_frames, fit.normal, fit.d)
        platform_mask = self._largest_plane_component(stable_mask)
        return fit, platform_mask

    def _stable_plane_mask(
        self, pointcloud_frames: Sequence[np.ndarray], normal: np.ndarray, d: float
    ) -> np.ndarray:
        """Pixels repeatedly within the plane threshold across empty frames."""

        masks: list[np.ndarray] = []
        for grid in pointcloud_frames:
            values = np.asarray(grid, dtype=np.float64)
            valid = np.isfinite(values).all(axis=2)
            residual = np.abs(point_to_plane_signed_distance(values, normal, d))
            masks.append(valid & (residual <= self.config.full_frame_plane_threshold_mm))
        return np.mean(np.stack(masks), axis=0) >= 0.5

    def _largest_plane_component(self, stable_mask: np.ndarray) -> np.ndarray:
        """Return the largest stable plane component without filling its holes.

        Morphological closing is used *only* to determine connectivity.  The
        returned support remains the original stable inlier mask, so a later
        safe ROI can never cross a no-depth hole or an obstacle-sized gap.
        """

        close_size = max(1, int(self.config.platform_roi_close_kernel_px))
        close_size = close_size if close_size % 2 else close_size + 1
        connected = cv2.morphologyEx(
            stable_mask.astype(np.uint8),
            cv2.MORPH_CLOSE,
            np.ones((close_size, close_size), np.uint8),
        )
        label_count, labels, stats, _ = cv2.connectedComponentsWithStats(connected, connectivity=8)
        if label_count <= 1:
            raise ValueError("could not find a connected platform-plane area")
        areas = stats[1:, cv2.CC_STAT_AREA]
        label = int(np.argmax(areas)) + 1
        component_area = int((stable_mask & (labels == label)).sum())
        if component_area < self.config.min_platform_component_pixels:
            raise ValueError(
                f"platform-plane area has {component_area} pixels; expected at least "
                f"{self.config.min_platform_component_pixels}"
            )
        return stable_mask & (labels == label)

    def _collect_masked_plane_inliers(
        self,
        pointcloud_frames: Sequence[np.ndarray],
        mask: np.ndarray,
        normal: np.ndarray,
        d: float,
    ) -> np.ndarray:
        """Collect full-frame plane points from the exact image-space support."""

        chunks: list[np.ndarray] = []
        for grid in pointcloud_frames:
            values = np.asarray(grid, dtype=np.float64)
            valid = np.isfinite(values).all(axis=2)
            residual = np.abs(point_to_plane_signed_distance(values, normal, d))
            inliers = valid & mask & (residual <= self.config.full_frame_plane_threshold_mm)
            frame_points = values[inliers]
            # Prevent a 640x480 x 60 calibration from needlessly retaining
            # millions of duplicate points before the final sample.
            chunks.append(sample_points(frame_points, self.config.max_sample_points))
        return np.concatenate(chunks) if chunks else np.empty((0, 3), dtype=np.float64)

    @staticmethod
    def _largest_safe_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
        """Find the largest axis-aligned rectangle made entirely of ``True`` pixels."""

        values = np.asarray(mask, dtype=bool)
        if values.ndim != 2:
            raise ValueError("platform mask must be a 2-D image")
        frame_height, frame_width = values.shape
        heights = np.zeros(frame_width, dtype=np.int32)
        best_area = 0
        best = (0, 0, 0, 0)
        for row in range(frame_height):
            heights = np.where(values[row], heights + 1, 0)
            stack: list[int] = []
            for column in range(frame_width + 1):
                current_height = int(heights[column]) if column < frame_width else 0
                while stack and current_height < int(heights[stack[-1]]):
                    top = stack.pop()
                    rectangle_height = int(heights[top])
                    left = stack[-1] + 1 if stack else 0
                    rectangle_width = column - left
                    area = rectangle_width * rectangle_height
                    if area > best_area:
                        best_area = area
                        best = (left, row - rectangle_height + 1, rectangle_width, rectangle_height)
                stack.append(column)
        if best_area == 0:
            raise ValueError("could not derive a safe rectangle inside the platform mask")
        return best

    def _roi_residual_statistics(
        self,
        pointcloud_frames: Sequence[np.ndarray],
        roi: tuple[int, int, int, int],
        normal: np.ndarray,
        d: float,
    ) -> tuple[float, float, float, float]:
        """Return signed residual p05/p50/p95 and standard deviation for the full ROI."""

        chunks: list[np.ndarray] = []
        for grid in pointcloud_frames:
            frame_points = extract_roi_points(grid, roi)
            chunks.append(sample_points(frame_points, self.config.max_sample_points))
        points = np.concatenate(chunks) if chunks else np.empty((0, 3), dtype=np.float64)
        if len(points) < self.config.min_inliers:
            raise ValueError(
                f"measurement ROI has only {len(points)} valid points; "
                f"expected at least {self.config.min_inliers}"
            )
        residuals = point_to_plane_signed_distance(points, normal, d)
        p05, p50, p95 = np.percentile(residuals, [5.0, 50.0, 95.0])
        return float(p05), float(p50), float(p95), float(np.std(residuals))

    def update_floor_for_measurement(
        self,
        current_points: np.ndarray,
        base_platform_model: PlatformModel,
        box_mask: np.ndarray | None = None,
    ) -> PlatformModel:
        """Robustly refit the live floor from the lower-height ROI points.

        At ``d`` time the platform can be a little tilted or shifted relative
        to its calibration frame.  Updating only ``d`` preserves that error as
        a false height gradient.  Here the low portion of the ROI is sampled
        and a constrained RANSAC/SVD fit updates both normal and ``d``.
        """

        grid = np.asarray(current_points)
        x, y, width, height = base_platform_model.measurement_roi
        roi_grid = grid[y : y + height, x : x + width].copy()
        if box_mask is not None:
            if box_mask.shape != grid.shape[:2]:
                raise ValueError("box_mask must match the point-cloud image size")
            roi_grid[box_mask[y : y + height, x : x + width]] = np.nan
        points = valid_points(roi_grid)
        if len(points) < self.config.dynamic_baseline_min_points:
            return base_platform_model
        base_residuals = point_to_plane_signed_distance(
            points, base_platform_model.plane_normal, base_platform_model.plane_d
        )
        low_cut, high_cut = np.percentile(
            base_residuals,
            [self.config.dynamic_floor_low_percentile, self.config.dynamic_floor_high_percentile],
        )
        floor_candidates = points[(base_residuals >= low_cut) & (base_residuals <= high_cut)]
        if len(floor_candidates) < self.config.dynamic_baseline_min_points:
            return base_platform_model
        try:
            live_fit = fit_plane_ransac(
                floor_candidates,
                iterations=self.config.dynamic_plane_iterations,
                threshold_mm=self.config.dynamic_plane_threshold_mm,
                min_inliers=self.config.dynamic_baseline_min_points,
                expected_normal=base_platform_model.plane_normal,
                min_normal_alignment=float(np.cos(np.deg2rad(self.config.dynamic_plane_max_tilt_deg))),
            )
        except ValueError:
            return base_platform_model
        if abs(live_fit.d - base_platform_model.plane_d) > self.config.dynamic_baseline_max_shift_mm:
            return base_platform_model
        p05, p50, p95, residual_std = self._point_residual_statistics(
            points, live_fit.normal, live_fit.d
        )
        return PlatformModel(
            frame_size=base_platform_model.frame_size,
            resolution=base_platform_model.resolution,
            plane_normal=live_fit.normal,
            plane_d=live_fit.d,
            camera_height_mm=abs(live_fit.d),
            measurement_roi=base_platform_model.measurement_roi,
            residual_std_mm=residual_std,
            created_at=base_platform_model.created_at,
            version=base_platform_model.version,
            platform_mask=base_platform_model.platform_mask,
            normal_alignment=base_platform_model.normal_alignment,
            roi_area_ratio=base_platform_model.roi_area_ratio,
            native_camera_height_mm=base_platform_model.native_camera_height_mm,
            applied_depth_scale=base_platform_model.applied_depth_scale,
            residual_p05_mm=p05,
            residual_p50_mm=p50,
            residual_p95_mm=p95,
        )

    @staticmethod
    def _point_residual_statistics(
        points: np.ndarray, normal: np.ndarray, d: float
    ) -> tuple[float, float, float, float]:
        residuals = point_to_plane_signed_distance(points, normal, d)
        p05, p50, p95 = np.percentile(residuals, [5.0, 50.0, 95.0])
        return float(p05), float(p50), float(p95), float(np.std(residuals))

"""Final SDK-independent box-volume pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from calibration.platform_calibrator import PlatformModel
from measurement.box_detector import BoxCandidate, BoxDetection, BoxDetector
from measurement.rectangle_fitter import RectangleFitter, RotatedRectangle, compute_width_length
from measurement.top_surface import TopSurfaceExtractor, TopSurfaceResult


@dataclass(frozen=True)
class VolumeResult:
    success: bool
    height_mm: float | None = None
    width_mm: float | None = None
    length_mm: float | None = None
    area_mm2: float | None = None
    volume_mm3: float | None = None
    volume_cm3: float | None = None
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)
    roi: tuple[int, int, int, int] | None = None
    box_point_count: int = 0
    top_point_count: int = 0
    top_plane_residual_std_mm: float | None = None
    candidate_mask: np.ndarray | None = field(default=None, repr=False, compare=False)
    candidates: tuple[BoxCandidate, ...] = ()


class BoxVolumeEstimator:
    """Compose detector → top surface → rectangle fitter into a result."""

    def __init__(
        self,
        measurement_config: object,
        box_detector: BoxDetector | None = None,
        top_surface_extractor: TopSurfaceExtractor | None = None,
        rectangle_fitter: RectangleFitter | None = None,
    ) -> None:
        self.config = measurement_config
        self.box_detector = box_detector or BoxDetector(measurement_config)
        self.top_surface_extractor = top_surface_extractor or TopSurfaceExtractor(measurement_config)
        self.rectangle_fitter = rectangle_fitter or RectangleFitter(measurement_config)

    def detect_box(self, points_grid: np.ndarray, platform_model: PlatformModel) -> BoxDetection:
        return self.box_detector.detect(points_grid, platform_model)

    def measure(self, points_grid: np.ndarray, platform_model: PlatformModel) -> VolumeResult:
        detection = self.detect_box(points_grid, platform_model)
        if not detection.found:
            return VolumeResult(
                False,
                warnings=[detection.warning or "box candidate was not found"],
                box_point_count=detection.point_count,
                candidate_mask=detection.candidate_mask,
                candidates=detection.candidates,
            )
        top = self.top_surface_extractor.extract(detection.points, detection.heights_mm, platform_model)
        if not top.success:
            return VolumeResult(
                False,
                warnings=[top.warning or "top surface is not stable"],
                roi=detection.roi,
                box_point_count=detection.point_count,
                top_point_count=len(top.points),
                top_plane_residual_std_mm=top.residual_std_mm,
                candidate_mask=detection.candidate_mask,
                candidates=detection.candidates,
            )
        try:
            rectangle = self.rectangle_fitter.fit(top.points, platform_model)
            return self._result(detection, top, rectangle)
        except ValueError as exc:
            return VolumeResult(
                False,
                warnings=[str(exc)],
                roi=detection.roi,
                box_point_count=detection.point_count,
                top_point_count=len(top.points),
                top_plane_residual_std_mm=top.residual_std_mm,
                candidate_mask=detection.candidate_mask,
                candidates=detection.candidates,
            )

    def _result(self, detection: BoxDetection, top: TopSurfaceResult, rectangle: RotatedRectangle) -> VolumeResult:
        assert top.top_height_mm is not None
        width, length = compute_width_length(rectangle)
        height = top.top_height_mm
        area = width * length
        volume = area * height
        residual_factor = max(0.0, 1.0 - (top.residual_std_mm or 0.0) / self.config.max_top_plane_residual_std_mm)
        point_factor = min(1.0, len(top.points) / (self.config.min_top_points * 4))
        confidence = float(np.clip(top.inlier_ratio * residual_factor * point_factor, 0.0, 1.0))
        return VolumeResult(
            True,
            height_mm=height,
            width_mm=width,
            length_mm=length,
            area_mm2=area,
            volume_mm3=volume,
            volume_cm3=volume / 1_000.0,
            confidence=confidence,
            roi=detection.roi,
            box_point_count=detection.point_count,
            top_point_count=len(top.points),
            top_plane_residual_std_mm=top.residual_std_mm,
            candidate_mask=detection.candidate_mask,
            candidates=detection.candidates,
        )

"""Detect elevated connected components above a calibrated platform plane."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from calibration.platform_calibrator import PlatformModel
from processing.plane_fitting import point_to_plane_signed_distance


@dataclass(frozen=True)
class BoxDetection:
    found: bool
    mask: np.ndarray
    roi: tuple[int, int, int, int] | None
    point_count: int
    points: np.ndarray
    heights_mm: np.ndarray


class BoxDetector:
    """Find the largest elevated component within the calibrated measurement ROI."""

    def __init__(self, measurement_config: object) -> None:
        self.config = measurement_config

    def detect(self, points_grid: np.ndarray, platform_model: PlatformModel) -> BoxDetection:
        grid = np.asarray(points_grid, dtype=np.float32)
        if grid.ndim != 3 or grid.shape[2] != 3:
            raise ValueError("points_grid must have shape H x W x 3")
        x, y, width, height = platform_model.measurement_roi
        if x < 0 or y < 0 or x + width > grid.shape[1] or y + height > grid.shape[0]:
            raise ValueError("platform measurement ROI is outside the point-cloud image")
        distances = point_to_plane_signed_distance(grid, platform_model.plane_normal, platform_model.plane_d)
        valid = np.isfinite(grid).all(axis=2)
        candidate = valid & (distances >= self.config.min_box_height_mm)
        full_mask = np.zeros(grid.shape[:2], dtype=bool)
        roi_mask = candidate[y : y + height, x : x + width].astype(np.uint8)
        if self.config.box_close_kernel > 1:
            size = self.config.box_close_kernel | 1
            roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, np.ones((size, size), np.uint8))
            # Never promote invalid pixels purely because the mask was closed.
            roi_mask &= valid[y : y + height, x : x + width].astype(np.uint8)
        labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(roi_mask, connectivity=8)
        if labels_count <= 1:
            return BoxDetection(False, full_mask, None, 0, np.empty((0, 3)), np.empty(0))
        areas = stats[1:, cv2.CC_STAT_AREA]
        label = int(np.argmax(areas)) + 1
        component = labels == label
        count = int(component.sum())
        if count < self.config.min_box_points:
            return BoxDetection(False, full_mask, None, count, np.empty((0, 3)), np.empty(0))
        full_mask[y : y + height, x : x + width] = component
        left = int(stats[label, cv2.CC_STAT_LEFT]) + x
        top = int(stats[label, cv2.CC_STAT_TOP]) + y
        box_width = int(stats[label, cv2.CC_STAT_WIDTH])
        box_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        points = grid[full_mask]
        heights = distances[full_mask]
        return BoxDetection(True, full_mask, (left, top, box_width, box_height), count, points, heights)

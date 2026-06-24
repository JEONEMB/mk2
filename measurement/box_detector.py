"""Detect elevated connected components above a calibrated platform plane."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from calibration.platform_calibrator import PlatformModel
from processing.plane_fitting import point_to_plane_signed_distance


@dataclass(frozen=True)
class BoxCandidate:
    roi: tuple[int, int, int, int]
    point_count: int
    median_height_mm: float
    max_height_mm: float
    accepted: bool
    reason: str | None = None


@dataclass(frozen=True)
class BoxDetection:
    found: bool
    mask: np.ndarray
    roi: tuple[int, int, int, int] | None
    point_count: int
    points: np.ndarray
    heights_mm: np.ndarray
    warning: str | None = None
    candidates: tuple[BoxCandidate, ...] = ()
    candidate_mask: np.ndarray | None = None


class BoxDetector:
    """Find elevated components within the calibrated rectangular measurement ROI."""

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
        roi_region = np.zeros(grid.shape[:2], dtype=bool)
        roi_region[y : y + height, x : x + width] = True
        # ``platform_model`` is dynamically re-centred at d time.  A candidate
        # is therefore any valid live point above that baseline inside the
        # rectangular placement ROI; fragmented calibration masks cannot carve
        # holes through a real box.
        candidate = valid & roi_region & (distances >= self.config.min_box_height_mm)
        full_mask = np.zeros(grid.shape[:2], dtype=bool)
        roi_mask = candidate[y : y + height, x : x + width].astype(np.uint8)
        if self.config.box_close_kernel > 1:
            size = self.config.box_close_kernel | 1
            roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, np.ones((size, size), np.uint8))
            # Never promote invalid pixels purely because the mask was closed.
            roi_mask &= valid[y : y + height, x : x + width].astype(np.uint8)
        raw_candidate_mask = np.zeros(grid.shape[:2], dtype=bool)
        raw_candidate_mask[y : y + height, x : x + width] = roi_mask.astype(bool)
        labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(roi_mask, connectivity=8)
        if labels_count <= 1:
            return BoxDetection(
                False,
                full_mask,
                None,
                0,
                np.empty((0, 3)),
                np.empty(0),
                "no elevated box candidate",
                candidate_mask=raw_candidate_mask,
            )
        labels_by_area = sorted(
            range(1, labels_count), key=lambda label: int(stats[label, cv2.CC_STAT_AREA]), reverse=True
        )
        label: int | None = None
        component: np.ndarray | None = None
        rejected_at_boundary = False
        candidate_stats: list[BoxCandidate] = []
        for candidate_label in labels_by_area:
            candidate_component = labels == candidate_label
            point_count = int(candidate_component.sum())
            candidate_full = np.zeros_like(full_mask)
            candidate_full[y : y + height, x : x + width] = candidate_component
            candidate_heights = distances[candidate_full]
            median_height = float(np.median(candidate_heights)) if len(candidate_heights) else 0.0
            max_height = float(np.max(candidate_heights)) if len(candidate_heights) else 0.0
            left = int(stats[candidate_label, cv2.CC_STAT_LEFT]) + x
            top = int(stats[candidate_label, cv2.CC_STAT_TOP]) + y
            box_width = int(stats[candidate_label, cv2.CC_STAT_WIDTH])
            box_height = int(stats[candidate_label, cv2.CC_STAT_HEIGHT])
            candidate_roi = (left, top, box_width, box_height)
            reason: str | None = None
            if point_count < self.config.min_box_points:
                reason = f"too few points (< {self.config.min_box_points})"
            dilated = cv2.dilate(candidate_full.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            if reason is None and np.any(dilated & ~roi_region):
                rejected_at_boundary = True
                reason = "touches measurement ROI boundary"
            accepted = reason is None
            candidate_stats.append(
                BoxCandidate(candidate_roi, point_count, median_height, max_height, accepted, reason)
            )
            if accepted and label is None:
                label, component, full_mask = candidate_label, candidate_component, candidate_full
        if label is None or component is None:
            warning = "box candidate touches the measurement ROI boundary" if rejected_at_boundary else "box candidate is too small"
            return BoxDetection(
                False,
                full_mask,
                None,
                0,
                np.empty((0, 3)),
                np.empty(0),
                warning,
                tuple(candidate_stats),
                raw_candidate_mask,
            )
        count = int(component.sum())
        left = int(stats[label, cv2.CC_STAT_LEFT]) + x
        top = int(stats[label, cv2.CC_STAT_TOP]) + y
        box_width = int(stats[label, cv2.CC_STAT_WIDTH])
        box_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        points = grid[full_mask]
        heights = distances[full_mask]
        return BoxDetection(
            True,
            full_mask,
            (left, top, box_width, box_height),
            count,
            points,
            heights,
            None,
            tuple(candidate_stats),
            raw_candidate_mask,
        )

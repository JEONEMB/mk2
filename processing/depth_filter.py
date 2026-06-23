"""SDK-free depth cleanup: invalid values, IR glare, spatial and temporal noise."""

from __future__ import annotations

from collections import deque
from typing import Iterable

import cv2
import numpy as np


def make_ir_saturation_mask(ir_image: np.ndarray | None, threshold: int, dilate_px: int = 0) -> np.ndarray | None:
    """Return pixels invalidated by saturated IR illumination."""

    if ir_image is None:
        return None
    ir = np.asarray(ir_image)
    if ir.ndim == 3:
        ir = np.max(ir, axis=2)
    mask = (ir >= threshold).astype(np.uint8)
    if dilate_px > 0:
        size = 2 * dilate_px + 1
        mask = cv2.dilate(mask, np.ones((size, size), np.uint8), iterations=1)
    return mask.astype(bool)


def remove_invalid_depth(depth_mm: np.ndarray, min_mm: float, max_mm: float) -> np.ndarray:
    """Convert every non-finite or out-of-range depth to ``NaN``."""

    clean = np.asarray(depth_mm, dtype=np.float32).copy()
    clean[~np.isfinite(clean) | (clean < min_mm) | (clean > max_mm)] = np.nan
    return clean


def apply_ir_mask(depth_mm: np.ndarray, ir_mask: np.ndarray | None) -> np.ndarray:
    clean = np.asarray(depth_mm, dtype=np.float32).copy()
    if ir_mask is not None:
        if clean.shape != ir_mask.shape:
            raise ValueError("IR mask and depth image must have the same shape")
        clean[ir_mask] = np.nan
    return clean


def median_filter_depth(depth_mm: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """NaN-aware median filter which does not turn missing pixels into zero."""

    depth = np.asarray(depth_mm, dtype=np.float32)
    if kernel_size <= 1:
        return depth.copy()
    if kernel_size % 2 == 0:
        raise ValueError("median kernel_size must be odd")
    padded = np.pad(depth, kernel_size // 2, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel_size, kernel_size))
    with np.errstate(all="ignore"):
        return np.nanmedian(windows, axis=(-2, -1)).astype(np.float32)


def temporal_median(depth_frames: Iterable[np.ndarray]) -> np.ndarray:
    frames = [np.asarray(frame, dtype=np.float32) for frame in depth_frames]
    if not frames:
        raise ValueError("at least one depth frame is required")
    shape = frames[0].shape
    if any(frame.shape != shape for frame in frames):
        raise ValueError("all depth frames must have identical dimensions")
    with np.errstate(all="ignore"):
        return np.nanmedian(np.stack(frames), axis=0).astype(np.float32)


def preprocess_depth(depth_mm: np.ndarray, ir_image: np.ndarray | None, depth_config: object) -> np.ndarray:
    """Perform one-frame preprocessing; temporal smoothing is stateful below."""

    cleaned = remove_invalid_depth(depth_mm, depth_config.min_valid_mm, depth_config.max_valid_mm)
    glare = make_ir_saturation_mask(
        ir_image, depth_config.ir_saturation_threshold, depth_config.ir_saturation_dilate_px
    )
    cleaned = apply_ir_mask(cleaned, glare)
    return median_filter_depth(cleaned, depth_config.spatial_median_kernel)


class DepthPreprocessor:
    """Keeps the temporal window separate from camera/SDK concerns."""

    def __init__(self, depth_config: object) -> None:
        self.config = depth_config
        self._history: deque[np.ndarray] = deque(maxlen=depth_config.temporal_window)

    def reset(self) -> None:
        self._history.clear()

    def process(self, depth_mm: np.ndarray, ir_image: np.ndarray | None = None) -> np.ndarray:
        spatial = preprocess_depth(depth_mm, ir_image, self.config)
        self._history.append(spatial)
        return temporal_median(self._history)

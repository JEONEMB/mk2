"""Linear depth-scale calibration from one or more known-distance references."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DepthScaleModel:
    scale: float = 1.0
    offset_mm: float = 0.0
    sample_count: int = 0
    rms_error_mm: float = 0.0

    def apply(self, depth_mm: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth_mm, dtype=np.float32)
        corrected = depth * self.scale + self.offset_mm
        corrected[~np.isfinite(depth)] = np.nan
        return corrected.astype(np.float32)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "scale": self.scale,
            "offset_mm": self.offset_mm,
            "sample_count": self.sample_count,
            "rms_error_mm": self.rms_error_mm,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DepthScaleModel":
        return cls(
            scale=float(data.get("scale", 1.0)),
            offset_mm=float(data.get("offset_mm", 0.0)),
            sample_count=int(data.get("sample_count", 0)),
            rms_error_mm=float(data.get("rms_error_mm", 0.0)),
        )


class DepthScaleCalibrator:
    """Stores (observed depth, known distance) pairs and fits known=a*observed+b."""

    def __init__(self) -> None:
        self._references: list[tuple[float, float]] = []

    def collect_reference(self, known_distance_mm: float, depth_mm: np.ndarray) -> float:
        values = np.asarray(depth_mm, dtype=np.float64)
        values = values[np.isfinite(values) & (values > 0)]
        if known_distance_mm <= 0 or not len(values):
            raise ValueError("known distance and the reference depth must be positive")
        observed = float(np.median(values))
        self._references.append((observed, float(known_distance_mm)))
        return observed

    def fit_scale_model(self) -> DepthScaleModel:
        if not self._references:
            raise ValueError("collect at least one reference before fitting depth scale")
        observed, known = np.asarray(self._references, dtype=np.float64).T
        if len(observed) == 1:
            scale, offset = known[0] / observed[0], 0.0
        else:
            scale, offset = np.linalg.lstsq(
                np.column_stack((observed, np.ones_like(observed))), known, rcond=None
            )[0]
        residual = known - (scale * observed + offset)
        return DepthScaleModel(float(scale), float(offset), len(observed), float(np.sqrt(np.mean(residual**2))))

    def apply(self, depth_mm: np.ndarray) -> np.ndarray:
        return self.fit_scale_model().apply(depth_mm)

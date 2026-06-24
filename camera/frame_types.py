"""SDK-independent data exchanged at the camera boundary."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics associated with a depth image."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def validate_for(self, shape: tuple[int, int]) -> None:
        if shape != (self.height, self.width):
            raise ValueError(
                f"intrinsics are {self.width}x{self.height}, but depth is {shape[1]}x{shape[0]}"
            )
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("camera focal lengths must be positive")

    def to_dict(self) -> dict[str, float | int]:
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class DepthFrame:
    """A single camera frame normalized to millimetre depth.

    No class in processing, calibration, or measurement needs to know which SDK
    produced this object.
    """

    depth_mm: np.ndarray
    intrinsics: CameraIntrinsics
    ir_image: np.ndarray | None = None
    pointcloud_xyz: np.ndarray | None = None
    timestamp: float = 0.0
    resolution_name: str = ""
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        depth = np.asarray(self.depth_mm)
        if depth.ndim != 2:
            raise ValueError("depth_mm must be a two-dimensional image")
        self.intrinsics.validate_for(depth.shape)
        if self.ir_image is not None and np.asarray(self.ir_image).shape[:2] != depth.shape:
            raise ValueError("IR and depth image sizes must match")
        if self.pointcloud_xyz is not None and np.asarray(self.pointcloud_xyz).shape != (*depth.shape, 3):
            raise ValueError("pointcloud_xyz must have shape H x W x 3 matching depth")

    @property
    def width(self) -> int:
        return self.depth_mm.shape[1]

    @property
    def height(self) -> int:
        return self.depth_mm.shape[0]

    def with_depth(self, depth_mm: np.ndarray, pointcloud_xyz: np.ndarray | None = None) -> "DepthFrame":
        """Return a new frame after scale correction/filtering."""

        return replace(
            self,
            depth_mm=np.asarray(depth_mm, dtype=np.float32),
            pointcloud_xyz=self.pointcloud_xyz if pointcloud_xyz is None else np.asarray(pointcloud_xyz, dtype=np.float32),
        )


@dataclass(frozen=True)
class CameraInfo:
    device_id: str
    model: str
    serial_number: str | None
    sdk_version: str | None

"""OpenCV display helpers. Visualisation does not participate in measurement."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from camera.frame_types import DepthFrame
    from measurement.volume_estimator import VolumeResult


class Viewer:
    def __init__(self, window_name: str = "Volume Scanner") -> None:
        self.window_name = window_name

    def show_depth(
        self,
        frame: "DepthFrame",
        *,
        depth_mm: np.ndarray | None = None,
        show_ir: bool = False,
        result: "VolumeResult | None" = None,
        platform_model: object | None = None,
    ) -> str:
        depth = frame.depth_mm if depth_mm is None else depth_mm
        if show_ir:
            if frame.ir_image is None:
                image = np.zeros((*depth.shape, 3), dtype=np.uint8)
                self._put_lines(image, ["IR frame is unavailable"])
            else:
                image = self._ir_image(frame.ir_image)
                self._put_lines(image, ["IR view  |  i: depth", "q: quit  c: calibrate  b: detect  d: measure"])
        else:
            image = self._depth_image(depth)
            self._put_lines(image, ["Depth view  |  i: IR", "q: quit  c: calibrate  b: detect  d: measure"])
        if platform_model is not None:
            self.draw_measurement_roi(image, platform_model.measurement_roi)
        if result is not None:
            if result.roi is not None:
                self.draw_box_roi(image, result.roi)
            self.draw_result_text(image, result)
        cv2.imshow(self.window_name, image)
        return self.read_key()

    @staticmethod
    def _depth_image(depth_mm: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth_mm, dtype=np.float32)
        valid = np.isfinite(depth)
        image = np.zeros(depth.shape, dtype=np.uint8)
        if valid.any():
            low, high = np.percentile(depth[valid], (2, 98))
            if high <= low:
                high = low + 1.0
            image[valid] = np.clip((depth[valid] - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
        return cv2.applyColorMap(image, cv2.COLORMAP_TURBO)

    @staticmethod
    def _ir_image(ir_image: np.ndarray) -> np.ndarray:
        ir = np.asarray(ir_image)
        if ir.ndim == 3:
            return ir.copy()
        return cv2.cvtColor(cv2.normalize(ir, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _put_lines(image: np.ndarray, lines: list[str], start_y: int = 24) -> None:
        for index, line in enumerate(lines):
            cv2.putText(image, line, (10, start_y + index * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(image, line, (10, start_y + index * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)

    def draw_measurement_roi(self, image: np.ndarray, roi: tuple[int, int, int, int]) -> None:
        x, y, width, height = roi
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 255), 1)

    def draw_box_roi(self, image: np.ndarray, roi: tuple[int, int, int, int]) -> None:
        x, y, width, height = roi
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

    def draw_result_text(self, image: np.ndarray, result: "VolumeResult") -> None:
        if result.success:
            lines = [
                f"H {result.height_mm:.1f} mm  W {result.width_mm:.1f} mm  L {result.length_mm:.1f} mm",
                f"Volume {result.volume_cm3:.1f} cm3  confidence {result.confidence:.2f}",
                f"top: {result.top_point_count} pts, std {result.top_plane_residual_std_mm:.2f} mm",
            ]
        else:
            lines = ["Measurement failed", *result.warnings]
        self._put_lines(image, lines, image.shape[0] - 62)

    @staticmethod
    def read_key() -> str:
        key = cv2.waitKey(1) & 0xFF
        return chr(key) if key != 255 else ""

    def close(self) -> None:
        cv2.destroyAllWindows()

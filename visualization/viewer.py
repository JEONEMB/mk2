"""OpenCV display helpers. Visualisation does not participate in measurement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from camera.frame_types import DepthFrame
    from measurement.box_detector import BoxDetection
    from measurement.volume_estimator import VolumeResult


@dataclass(frozen=True)
class OverlayStyle:
    """Typography sized for the source 320 x 240 depth image.

    OpenCV enlarges the window on desktop displays, so these values deliberately
    use half the previous source-image font size and line spacing.
    """

    text_scale: float = 0.275
    roi_scale: float = 0.25
    candidate_scale: float = 0.19
    line_height_px: int = 10
    margin_x_px: int = 5
    header_start_y_px: int = 12
    label_offset_y_px: int = 3
    outline_thickness: int = 1
    text_thickness: int = 1
    max_candidate_labels: int = 3


HEIGHT_VIEW_MIN_MM = -20.0
HEIGHT_VIEW_MAX_MM = 120.0
_INVALID_HEIGHT_COLOR = (18, 8, 18)


class Viewer:
    """Render depth by default and toggle an independent IR window with ``i``."""

    def __init__(self, depth_window: str = "Volume Scanner - Depth", ir_window: str = "Volume Scanner - IR") -> None:
        self.depth_window = depth_window
        self.ir_window = ir_window
        self._depth_window_created = False
        self._ir_window_created = False
        self.ir_visible = False
        self.style = OverlayStyle()

    def toggle_ir(self) -> bool:
        """Show or hide the independent IR window and return its new state."""

        self.ir_visible = not self.ir_visible
        if not self.ir_visible and self._ir_window_created:
            try:
                cv2.destroyWindow(self.ir_window)
            except cv2.error:
                pass
            self._ir_window_created = False
        return self.ir_visible

    def show_frames(
        self,
        frame: "DepthFrame",
        *,
        depth_mm: np.ndarray | None = None,
        result: "VolumeResult | None" = None,
        detection: "BoxDetection | None" = None,
        platform_model: object | None = None,
        diagnostics: dict[str, float] | None = None,
        height_above_platform_mm: np.ndarray | None = None,
        height_view_active: bool = False,
        dynamic_baseline_delta_mm: float | None = None,
    ) -> str:
        """Show the current depth and IR frames and return the pressed key."""

        depth = frame.depth_mm if depth_mm is None else np.asarray(depth_mm)
        has_height_view = height_view_active and height_above_platform_mm is not None
        if has_height_view:
            depth_image = self._height_image(height_above_platform_mm)
            lines = [
                f"Height above platform: {HEIGHT_VIEW_MIN_MM:.0f}..{HEIGHT_VIEW_MAX_MM:.0f} mm",
                "0mm=cold  +70mm=warm  h: depth view",
                "i: IR  r: resolution  c: calibrate  b: detect  d: measure  q: quit",
            ]
        else:
            depth_image = self._depth_image(depth)
            lines = [
                f"Depth: {depth.shape[1]} x {depth.shape[0]}",
                "h: height map  i: IR  r: resolution  c: calibrate  b: detect  d: measure  q: quit",
            ]
        if diagnostics is not None:
            lines.append(
                f"valid: {diagnostics.get('valid_depth_ratio', 0.0):.1%}  "
                f"IR masked: {diagnostics.get('ir_mask_ratio', 0.0):.1%}"
            )
        if has_height_view and dynamic_baseline_delta_mm is not None:
            lines.append(f"dynamic baseline: {dynamic_baseline_delta_mm:+.1f} mm")
        self._put_lines(depth_image, lines)
        if platform_model is not None:
            self.draw_measurement_roi(
                depth_image,
                platform_model.measurement_roi,
                getattr(platform_model, "camera_height_mm", None),
            )
        if result is not None:
            if result.candidate_mask is not None:
                self.draw_candidate_mask(depth_image, result.candidate_mask)
            self.draw_candidate_stats(depth_image, result.candidates)
            if result.roi is not None:
                self.draw_box_roi(depth_image, result.roi)
            self.draw_result_text(depth_image, result)
        elif detection is not None:
            if detection.candidate_mask is not None:
                self.draw_candidate_mask(depth_image, detection.candidate_mask)
            self.draw_candidate_stats(depth_image, detection.candidates)
            if detection.roi is not None:
                self.draw_box_roi(depth_image, detection.roi)
            self.draw_detection_text(depth_image, detection)

        self._create_depth_window()
        cv2.imshow(self.depth_window, depth_image)
        if self.ir_visible:
            if frame.ir_image is None:
                ir_image = np.zeros_like(depth_image)
                self._put_lines(ir_image, ["IR: unavailable"])
            else:
                ir = np.asarray(frame.ir_image)
                ir_image = self._ir_image(ir)
                self._put_lines(ir_image, [f"IR: {ir.shape[1]} x {ir.shape[0]}"])
            self._create_ir_window(depth_image.shape[1])
            cv2.imshow(self.ir_window, ir_image)
        return self.read_key()

    def show_depth(
        self,
        frame: "DepthFrame",
        *,
        depth_mm: np.ndarray | None = None,
        result: "VolumeResult | None" = None,
        detection: "BoxDetection | None" = None,
        platform_model: object | None = None,
        diagnostics: dict[str, float] | None = None,
        height_above_platform_mm: np.ndarray | None = None,
        height_view_active: bool = False,
        dynamic_baseline_delta_mm: float | None = None,
        **_: object,
    ) -> str:
        """Backward-compatible alias; depth and IR are always shown together."""

        return self.show_frames(
            frame,
            depth_mm=depth_mm,
            result=result,
            detection=detection,
            platform_model=platform_model,
            diagnostics=diagnostics,
            height_above_platform_mm=height_above_platform_mm,
            height_view_active=height_view_active,
            dynamic_baseline_delta_mm=dynamic_baseline_delta_mm,
        )

    def _create_depth_window(self) -> None:
        if self._depth_window_created:
            return
        cv2.namedWindow(self.depth_window, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.depth_window, 40, 40)
        self._depth_window_created = True

    def _create_ir_window(self, depth_width: int) -> None:
        if self._ir_window_created:
            return
        cv2.namedWindow(self.ir_window, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.ir_window, 60 + depth_width, 40)
        self._ir_window_created = True

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
    def _height_image(height_mm: np.ndarray) -> np.ndarray:
        """Render a fixed platform-relative height range for measurement diagnosis."""

        heights = np.asarray(height_mm, dtype=np.float32)
        valid = np.isfinite(heights)
        image = np.zeros(heights.shape, dtype=np.uint8)
        image[valid] = np.clip(
            (heights[valid] - HEIGHT_VIEW_MIN_MM)
            * 255.0
            / (HEIGHT_VIEW_MAX_MM - HEIGHT_VIEW_MIN_MM),
            0,
            255,
        ).astype(np.uint8)
        rendered = cv2.applyColorMap(image, cv2.COLORMAP_TURBO)
        rendered[~valid] = _INVALID_HEIGHT_COLOR
        return rendered

    @staticmethod
    def _ir_image(ir_image: np.ndarray) -> np.ndarray:
        ir = np.asarray(ir_image)
        if ir.ndim == 3:
            return ir.copy()
        normalized = cv2.normalize(ir, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)

    def _put_lines(self, image: np.ndarray, lines: list[str], start_y: int | None = None) -> None:
        y_start = self.style.header_start_y_px if start_y is None else start_y
        for index, line in enumerate(lines):
            y = y_start + index * self.style.line_height_px
            position = (self.style.margin_x_px, y)
            cv2.putText(
                image,
                line,
                position,
                cv2.FONT_HERSHEY_SIMPLEX,
                self.style.text_scale,
                (30, 30, 30),
                self.style.outline_thickness,
            )
            cv2.putText(
                image,
                line,
                position,
                cv2.FONT_HERSHEY_SIMPLEX,
                self.style.text_scale,
                (255, 255, 255),
                self.style.text_thickness,
            )

    def draw_measurement_roi(self, image: np.ndarray, roi: tuple[int, int, int, int], height_mm: float | None = None) -> None:
        x, y, width, height = roi
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 255), 1)
        label = "Platform ROI" if height_mm is None else f"Platform ROI  H {height_mm:.1f} mm"
        cv2.putText(
            image,
            label,
            (x, max(9, y - self.style.label_offset_y_px)),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.style.roi_scale,
            (0, 255, 255),
            self.style.text_thickness,
        )

    def draw_box_roi(self, image: np.ndarray, roi: tuple[int, int, int, int]) -> None:
        x, y, width, height = roi
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

    @staticmethod
    def draw_candidate_mask(image: np.ndarray, candidate_mask: np.ndarray) -> None:
        """Tint every elevated candidate so false positives remain visible."""

        mask = np.asarray(candidate_mask, dtype=bool)
        if mask.shape != image.shape[:2]:
            return
        overlay = image.copy()
        overlay[mask] = (0, 180, 255)
        image[:] = cv2.addWeighted(overlay, 0.45, image, 0.55, 0.0)

    def draw_candidate_stats(self, image: np.ndarray, candidates: tuple[object, ...]) -> None:
        """Draw the selected candidate and at most two large rejected candidates."""

        for index, candidate in self._display_candidates(candidates):
            x, y, width, height = candidate.roi
            color = (0, 255, 0) if candidate.accepted else (0, 80, 255)
            cv2.rectangle(image, (x, y), (x + width, y + height), color, 1)
            label = f"C{index} H{candidate.median_height_mm:.1f} P{candidate.point_count}"
            label_y = max(7, y - self.style.label_offset_y_px)
            cv2.putText(
                image,
                label,
                (x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.style.candidate_scale,
                (0, 0, 0),
                self.style.outline_thickness,
            )
            cv2.putText(
                image,
                label,
                (x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.style.candidate_scale,
                color,
                self.style.text_thickness,
            )

    def _display_candidates(self, candidates: tuple[object, ...]) -> list[tuple[int, object]]:
        """Keep diagnostics useful without letting tiny rejected blobs cover the view."""

        indexed = list(enumerate(candidates, start=1))
        selected = next(((index, candidate) for index, candidate in indexed if candidate.accepted), None)
        rejected = [(index, candidate) for index, candidate in indexed if not candidate.accepted]
        if selected is None:
            return rejected[: self.style.max_candidate_labels]
        return [selected, *rejected[: self.style.max_candidate_labels - 1]]

    def draw_result_text(self, image: np.ndarray, result: "VolumeResult") -> None:
        if result.success:
            lines = [
                f"H {result.height_mm:.1f} mm  W {result.width_mm:.1f} mm  L {result.length_mm:.1f} mm",
                f"Volume {result.volume_cm3:.1f} cm3  confidence {result.confidence:.2f}",
                f"top: {result.top_point_count} pts, std {result.top_plane_residual_std_mm:.2f} mm",
            ]
        else:
            lines = ["Measurement failed", *result.warnings, "top requires >= 300 points"]
        self._put_lines(image, lines, max(10, image.shape[0] - 31))

    def draw_detection_text(self, image: np.ndarray, detection: "BoxDetection") -> None:
        if detection.found:
            lines = [f"Detection: {detection.point_count} candidate points"]
        else:
            lines = ["Detection rejected", detection.warning or "no elevated candidate"]
        self._put_lines(image, lines, max(10, image.shape[0] - 21))

    @staticmethod
    def read_key() -> str:
        key = cv2.waitKey(1) & 0xFF
        return chr(key) if key != 255 else ""

    def close(self) -> None:
        cv2.destroyAllWindows()
        self._depth_window_created = False
        self._ir_window_created = False

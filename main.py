"""Application composition root for the CS20 volume scanner.

This file intentionally coordinates components only.  Synexens SDK calls stay
inside ``camera/synexens_camera.py``; all measurement code works with clean
millimetre arrays and calibration models.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import numpy as np

from calibration.calibration_store import (
    load_depth_scale_model,
    load_platform_model,
    save_depth_scale_model,
    save_platform_model,
)
from calibration.depth_scale_calibrator import DepthScaleCalibrator, DepthScaleModel
from calibration.platform_calibrator import PlaneProposal, PlatformCalibrator, PlatformModel
from camera.frame_types import DepthFrame
from camera.synexens_camera import CameraError, SynexensCamera, SyntheticCameraBackend
from config import CalibrationConfig, DepthConfig, MeasurementConfig, SDKConfig
from measurement.box_detector import BoxDetector
from measurement.rectangle_fitter import RectangleFitter
from measurement.top_surface import TopSurfaceExtractor
from measurement.volume_estimator import BoxVolumeEstimator, VolumeResult
from processing.depth_filter import DepthPreprocessor
from processing.plane_fitting import point_to_plane_signed_distance
from processing.pointcloud_utils import depth_to_pointcloud, rescale_pointcloud_to_depth
from visualization.viewer import Viewer


_MIN_REASONABLE_DEPTH_SCALE = 0.5
_MAX_REASONABLE_DEPTH_SCALE = 2.0
_INTERACTIVE_RESOLUTIONS = ("320x240", "640x480")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CS20 box-volume scanner")
    parser.add_argument("--demo", action="store_true", help="use a deterministic synthetic CS20 frame source")
    parser.add_argument("--headless", action="store_true", help="do not open an OpenCV window")
    parser.add_argument("--calibrate", action="store_true", help="save an empty-platform model and exit")
    parser.add_argument("--detect", action="store_true", help="print only box point count and ROI, then exit")
    parser.add_argument("--measure", action="store_true", help="perform one measurement and exit")
    parser.add_argument(
        "--platform-height-mm",
        type=float,
        help="known lens-center-to-platform distance in mm, used with --calibrate",
    )
    parser.add_argument(
        "--depth-reference-mm",
        type=float,
        help="fit and save a one-reference depth-scale model using the next frame",
    )
    return parser.parse_args()


def clean_frame(
    frame: DepthFrame, preprocessor: DepthPreprocessor, scale_model: DepthScaleModel | None
) -> tuple[DepthFrame, np.ndarray]:
    """Apply the same calibrated depth to both display and point-cloud conversion."""

    scaled = scale_model.apply(frame.depth_mm) if scale_model else frame.depth_mm
    cleaned_depth = preprocessor.process(scaled, frame.ir_image)
    if frame.pointcloud_xyz is not None:
        cloud = rescale_pointcloud_to_depth(frame.pointcloud_xyz, cleaned_depth)
    else:
        cloud = depth_to_pointcloud(cleaned_depth, frame.intrinsics)
    cleaned_frame = frame.with_depth(cleaned_depth, cloud)
    raw_cloud = (frame.metadata or {}).get("raw_pointcloud_xyz")
    if raw_cloud is not None:
        metadata = dict(frame.metadata or {})
        # Preserve the vendor's non-undistorted geometry as an A/B diagnostic.
        # It is intentionally not rescaled onto the undistorted rays used for
        # measurement, otherwise the comparison would hide distortion.
        raw_geometry = np.asarray(raw_cloud, dtype=np.float32).copy()
        raw_geometry[~np.isfinite(cleaned_depth)] = np.nan
        metadata["raw_pointcloud_xyz"] = raw_geometry
        cleaned_frame = replace(cleaned_frame, metadata=metadata)
    return cleaned_frame, cloud


@dataclass(frozen=True)
class PlatformCapture:
    points: list[np.ndarray]
    raw_points: list[np.ndarray] | None
    resolution: str
    preview_depth_mm: np.ndarray
    proposal: PlaneProposal


def collect_pointclouds(
    camera: SynexensCamera,
    count: int,
    preprocessor: DepthPreprocessor,
    scale_model: DepthScaleModel | None,
) -> tuple[list[np.ndarray], list[np.ndarray] | None, str, np.ndarray]:
    points: list[np.ndarray] = []
    raw_points: list[np.ndarray] = []
    raw_available = True
    resolution = ""
    preview_depth = np.empty((0, 0), dtype=np.float32)
    while len(points) < count:
        frame = camera.read()
        if frame is None:
            continue
        cleaned, cloud = clean_frame(frame, preprocessor, scale_model)
        resolution = cleaned.resolution_name
        preview_depth = cleaned.depth_mm
        points.append(cloud)
        raw_cloud = (cleaned.metadata or {}).get("raw_pointcloud_xyz")
        if raw_cloud is None:
            raw_available = False
        else:
            raw_points.append(np.asarray(raw_cloud, dtype=np.float32))
    return points, raw_points if raw_available else None, resolution, preview_depth


def capture_platform_proposal(
    camera: SynexensCamera,
    calibrator: PlatformCalibrator,
    config: CalibrationConfig,
    preprocessor: DepthPreprocessor,
) -> PlatformCapture:
    """Capture one empty-platform frame set and derive its unsaved proposal."""

    preprocessor.reset()
    points, raw_points, resolution, preview_depth = collect_pointclouds(
        camera, config.frame_count, preprocessor, None
    )
    return PlatformCapture(
        points, raw_points, resolution, preview_depth, calibrator.propose_platform(points, raw_points)
    )


def make_estimator(config: MeasurementConfig) -> BoxVolumeEstimator:
    """Wire the final detector → top surface → rectangle pipeline."""

    detector = BoxDetector(config)
    top_surface = TopSurfaceExtractor(config)
    rectangle = RectangleFitter(config)
    return BoxVolumeEstimator(config, detector, top_surface, rectangle)


def print_result(result: VolumeResult) -> None:
    data = asdict(result)
    # The image-sized mask belongs on the OpenCV overlay, not in the terminal.
    data.pop("candidate_mask", None)
    print(data)


def candidate_summary(candidates: tuple[object, ...]) -> list[dict[str, object]]:
    """Keep terminal diagnostics compact while exposing every connected component."""

    return [
        {
            "roi": candidate.roi,
            "height_mm": round(candidate.median_height_mm, 2),
            "max_height_mm": round(candidate.max_height_mm, 2),
            "point_count": candidate.point_count,
            "accepted": candidate.accepted,
            "reason": candidate.reason,
        }
        for candidate in candidates
    ]


def plane_quality_summary(quality: object | None) -> dict[str, float] | None:
    if quality is None:
        return None
    return {
        "p05_mm": round(float(quality.p05_mm), 2),
        "p50_mm": round(float(quality.p50_mm), 2),
        "p95_mm": round(float(quality.p95_mm), 2),
        "std_mm": round(float(quality.std_mm), 2),
    }


def proposal_summary(proposal: PlaneProposal) -> dict[str, object]:
    """Keep preview diagnostics numeric and explicit before any state is saved."""

    return {
        "accepted": proposal.accepted,
        "outer_roi": proposal.outer_roi,
        "measurement_roi": proposal.measurement_roi,
        "component_area_px": proposal.component_area_px,
        "support_ratio": round(proposal.support_ratio, 4),
        "roi_coverage": round(proposal.roi_coverage, 4) if proposal.roi_coverage is not None else None,
        "normal_alignment": round(proposal.normal_alignment, 4),
        "undistorted_quality_mm": plane_quality_summary(proposal.quality),
        "raw_quality_mm": plane_quality_summary(proposal.raw_quality),
        "failure_reason": proposal.failure_reason,
    }


def print_detection(camera: SynexensCamera, estimator: BoxVolumeEstimator, preprocessor: DepthPreprocessor,
                    scale_model: DepthScaleModel | None, platform_model: PlatformModel | None) -> int:
    if platform_model is None:
        print("Detection skipped: platform_plane.json is not available. Run --calibrate first.")
        return 2
    frame = _next_frame(camera)
    _, cloud = clean_frame(frame, preprocessor, scale_model)
    detection = estimator.detect_box(cloud, platform_model)
    print(
        {
            "found": detection.found,
            "box_point_count": detection.point_count,
            "roi": detection.roi,
            "warning": detection.warning,
            "candidates": candidate_summary(detection.candidates),
        }
    )
    return 0 if detection.found else 1


def calibrate_platform(
    camera: SynexensCamera,
    calibrator: PlatformCalibrator,
    config: CalibrationConfig,
    preprocessor: DepthPreprocessor,
    reference_height_mm: float | None = None,
    capture: PlatformCapture | None = None,
) -> tuple[PlatformModel, DepthScaleModel | None]:
    """Persist only a confirmed proposal produced from this exact frame set."""

    captured = capture or capture_platform_proposal(camera, calibrator, config, preprocessor)
    proposal = captured.proposal
    if not proposal.accepted or proposal.outer_roi is None:
        raise ValueError(f"platform proposal rejected: {proposal.failure_reason or 'no safe measurement ROI'}")
    # The proposal was created from native SDK geometry. A ruler reference
    # replaces its scale rather than compounding a prior calibration.
    raw_model = calibrator.calibrate(captured.points, resolution=captured.resolution, manual_roi=proposal.outer_roi)
    scale_model: DepthScaleModel | None = None
    model = raw_model
    if reference_height_mm is not None:
        if reference_height_mm <= 0:
            raise ValueError("measured camera height must be positive")
        correction = reference_height_mm / raw_model.camera_height_mm
        if not _MIN_REASONABLE_DEPTH_SCALE <= correction <= _MAX_REASONABLE_DEPTH_SCALE:
            raise ValueError(
                f"height scale {correction:.4f} is implausible. Check the entered height; "
                "enter 700 or 70cm for a seventy-centimetre installation."
            )
        scale_model = DepthScaleModel(
            scale=correction,
            offset_mm=0.0,
            sample_count=1,
            rms_error_mm=abs(reference_height_mm - raw_model.camera_height_mm),
        )
        scaled_points = [point_grid * correction for point_grid in captured.points]
        model = calibrator.calibrate(
            scaled_points, resolution=captured.resolution, manual_roi=proposal.outer_roi
        )
        model = replace(
            model,
            native_camera_height_mm=raw_model.camera_height_mm,
            applied_depth_scale=correction,
        )
        save_depth_scale_model(scale_model)
        print(
            "Camera height calibration: "
            f"SDK plane={raw_model.camera_height_mm:.2f} mm, "
            f"measured={reference_height_mm:.2f} mm, scale={correction:.8f}, "
            f"corrected={model.camera_height_mm:.2f} mm"
        )
    path = save_platform_model(model)
    print(
        f"Platform plane saved to {path}; camera_height_mm={model.camera_height_mm:.2f}; "
        f"normal alignment={model.normal_alignment:.3f}; ROI area={model.roi_area_ratio:.1%}; "
        f"platform mask={int(model.platform_mask.sum()) if model.platform_mask is not None else 0} px"
    )
    print(
        "Measurement ROI plane residuals (mm): "
        f"p05={model.residual_p05_mm:.2f}, p50={model.residual_p50_mm:.2f}, "
        f"p95={model.residual_p95_mm:.2f}, std={model.residual_std_mm:.2f}"
    )
    return model, scale_model


def parse_platform_height_mm(response: str) -> float:
    """Convert ``700``, ``70cm``, or bare ``70`` to a positive millimetre height."""

    value = response.strip().lower()
    multiplier = 1.0
    if value.endswith("cm"):
        value = value[:-2].strip()
        multiplier = 10.0
    try:
        height_mm = float(value) * multiplier
    except ValueError as exc:
        raise ValueError("Enter a positive height, for example 700 or 70cm.") from exc
    # Camera-to-platform distances below 100 mm are implausible here. Most
    # operators naturally enter '70' for 70 cm, so interpret it safely.
    if multiplier == 1.0 and 0 < height_mm < 100.0:
        height_mm *= 10.0
    if height_mm <= 0:
        raise ValueError("Height must be greater than zero.")
    return height_mm


def prompt_platform_height_mm() -> float | None:
    """Show a small modal dialog for the measured lens-to-platform height.

    ``None`` means that the operator pressed Cancel, which must not overwrite
    an existing calibration.
    """

    try:
        import tkinter as tk
        from tkinter import messagebox, simpledialog
    except ImportError as exc:  # pragma: no cover - standard Windows Python includes Tk.
        raise ValueError("Tkinter is required to show the camera-height input window.") from exc

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise ValueError(f"Could not open the camera-height input window: {exc}") from exc
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        while True:
            response = simpledialog.askstring(
                "Camera height",
                "렌즈 중심에서 빈 플랫폼까지의 거리\n"
                "mm 단위로 입력하세요.  예: 700 또는 70cm\n"
                "숫자 70은 70cm(700mm)로 해석합니다.",
                initialvalue="700",
                parent=root,
            )
            if response is None:
                return None
            try:
                return parse_platform_height_mm(response)
            except ValueError as exc:
                messagebox.showerror("입력 오류", str(exc), parent=root)
    finally:
        root.destroy()


def calibrate_depth_scale(camera: SynexensCamera, known_distance_mm: float) -> DepthScaleModel:
    frame = _next_frame(camera)
    calibrator = DepthScaleCalibrator()
    observed = calibrator.collect_reference(known_distance_mm, frame.depth_mm)
    model = calibrator.fit_scale_model()
    path = save_depth_scale_model(model)
    print(f"Depth scale saved to {path}; observed={observed:.2f} mm, scale={model.scale:.8f}")
    return model


def _next_frame(camera: SynexensCamera) -> DepthFrame:
    while True:
        frame = camera.read()
        if frame is not None:
            return frame


def run_once_measurement(
    camera: SynexensCamera,
    estimator: BoxVolumeEstimator,
    preprocessor: DepthPreprocessor,
    scale_model: DepthScaleModel | None,
    platform_model: PlatformModel | None,
) -> tuple[DepthFrame, np.ndarray, VolumeResult | None]:
    frame = _next_frame(camera)
    cleaned, cloud = clean_frame(frame, preprocessor, scale_model)
    if platform_model is None:
        return cleaned, cloud, None
    return cleaned, cloud, estimator.measure(cloud, platform_model)


def measure_with_dynamic_baseline(
    estimator: BoxVolumeEstimator,
    calibrator: PlatformCalibrator,
    points_grid: np.ndarray,
    platform_model: PlatformModel,
) -> tuple[VolumeResult, PlatformModel]:
    """Use the d-time floor offset before selecting elevated box points."""

    dynamic_model = calibrator.update_floor_for_measurement(points_grid, platform_model)
    return estimator.measure(points_grid, dynamic_model), dynamic_model


def platform_relative_height_map(points_grid: np.ndarray, platform_model: PlatformModel) -> np.ndarray:
    """Return live height above the platform inside its calibrated rectangular ROI."""

    heights = point_to_plane_signed_distance(
        points_grid, platform_model.plane_normal, platform_model.plane_d
    ).astype(np.float32)
    x, y, width, height = platform_model.measurement_roi
    roi_heights = np.full(heights.shape, np.nan, dtype=np.float32)
    roi_heights[y : y + height, x : x + width] = heights[y : y + height, x : x + width]
    return roi_heights


def next_interactive_resolution(current_resolution: str) -> str:
    """Toggle between the two CS20 resolutions used by the interactive viewer."""

    return "640x480" if current_resolution == "320x240" else "320x240"


def interactive_loop(
    camera: SynexensCamera,
    calibrator: PlatformCalibrator,
    estimator: BoxVolumeEstimator,
    calibration_config: CalibrationConfig,
    preprocessor: DepthPreprocessor,
    scale_model: DepthScaleModel | None,
    platform_model: PlatformModel | None,
) -> None:
    viewer = Viewer()
    result: VolumeResult | None = None
    detection_preview = None
    plane_preview = None
    height_view_active = False
    try:
        while True:
            cleaned, cloud, _ = run_once_measurement(camera, estimator, preprocessor, scale_model, None)
            current_resolution = cleaned.resolution_name
            dynamic_height_model: PlatformModel | None = None
            relative_height_mm: np.ndarray | None = None
            if height_view_active and platform_model is not None:
                dynamic_height_model = calibrator.update_floor_for_measurement(cloud, platform_model)
                relative_height_mm = platform_relative_height_map(cloud, dynamic_height_model)
            key = viewer.show_frames(
                cleaned,
                depth_mm=cleaned.depth_mm,
                result=result,
                detection=detection_preview,
                platform_model=platform_model,
                diagnostics=preprocessor.diagnostics,
                plane_preview=plane_preview,
                height_above_platform_mm=relative_height_mm,
                height_view_active=height_view_active,
                dynamic_baseline_delta_mm=(
                    dynamic_height_model.camera_height_mm - platform_model.camera_height_mm
                    if dynamic_height_model is not None and platform_model is not None
                    else None
                ),
            )
            key = key.lower()
            if key == "q":
                return
            if key == "i":
                print(f"IR window {'opened' if viewer.toggle_ir() else 'closed'}.")
            elif key == "h":
                if platform_model is None:
                    print("Height map needs an empty-platform calibration. Press c first.")
                else:
                    height_view_active = not height_view_active
                    print(f"Platform-relative height map {'opened' if height_view_active else 'closed'}.")
            elif key == "r":
                target_resolution = next_interactive_resolution(current_resolution)
                try:
                    camera.set_resolution(target_resolution)
                    preprocessor.reset()
                    # Pixel coordinates and intrinsics change with resolution.
                    # Keep the distance scale, but require a fresh empty-platform
                    # ROI/plane calibration before detecting or measuring.
                    platform_model = None
                    result = None
                    detection_preview = None
                    height_view_active = False
                    print(
                        f"Resolution changed: {current_resolution} -> {target_resolution}. "
                        "Press c to calibrate the new resolution."
                    )
                except CameraError as exc:
                    print(f"Resolution change failed: {exc}")
            elif key == "b":
                if platform_model is None:
                    print("No platform model. Press c with an empty platform first.")
                else:
                    detection = estimator.detect_box(cloud, platform_model)
                    detection_preview = detection
                    result = None
                    print(
                        {
                            "found": detection.found,
                            "box_point_count": detection.point_count,
                            "roi": detection.roi,
                            "warning": detection.warning,
                            "candidates": candidate_summary(detection.candidates),
                        }
                    )
            elif key == "p":
                try:
                    print("Capturing empty-platform frames for plane preview...")
                    capture = capture_platform_proposal(camera, calibrator, calibration_config, preprocessor)
                    plane_preview = capture.proposal
                    print(proposal_summary(plane_preview))
                except ValueError as exc:
                    plane_preview = None
                    print(f"Plane preview failed: {exc}")
            elif key == "c":
                try:
                    print("Capturing empty-platform frames for verified ROI proposal...")
                    capture = capture_platform_proposal(camera, calibrator, calibration_config, preprocessor)
                    plane_preview = capture.proposal
                    print(proposal_summary(plane_preview))
                    if not viewer.confirm_platform_proposal(capture.preview_depth_mm, plane_preview):
                        if not plane_preview.accepted:
                            print(f"Platform proposal rejected: {plane_preview.failure_reason}")
                        else:
                            print("Platform calibration cancelled; existing calibration was kept.")
                        continue
                    reference_height_mm = prompt_platform_height_mm()
                    if reference_height_mm is None:
                        print("Platform calibration cancelled; existing calibration was kept.")
                        continue
                    print(f"Calibrating verified platform ROI {plane_preview.measurement_roi}: keep it empty...")
                    platform_model, new_scale_model = calibrate_platform(
                        camera,
                        calibrator,
                        calibration_config,
                        preprocessor,
                        reference_height_mm,
                        capture,
                    )
                    scale_model = new_scale_model
                    result = None
                    detection_preview = None
                    plane_preview = None
                    height_view_active = False
                    print(
                        f"Platform ROI: {platform_model.measurement_roi}; "
                        f"camera_height_mm={platform_model.camera_height_mm:.2f}"
                    )
                except ValueError as exc:
                    print(f"Platform calibration failed: {exc}")
            elif key == "d":
                if platform_model is None:
                    print("No platform model. Press c with an empty platform first.")
                else:
                    result, dynamic_model = measure_with_dynamic_baseline(
                        estimator, calibrator, cloud, platform_model
                    )
                    detection_preview = None
                    print(
                        f"Dynamic baseline: {dynamic_model.camera_height_mm:.2f} mm "
                        f"(delta {dynamic_model.camera_height_mm - platform_model.camera_height_mm:+.2f} mm)"
                    )
                    print(
                        "Dynamic ROI plane residuals (mm): "
                        f"p05={dynamic_model.residual_p05_mm:.2f}, "
                        f"p50={dynamic_model.residual_p50_mm:.2f}, "
                        f"p95={dynamic_model.residual_p95_mm:.2f}, "
                        f"std={dynamic_model.residual_std_mm:.2f}"
                    )
                    print_result(result)
    finally:
        viewer.close()


def main() -> int:
    args = parse_args()
    sdk_config = SDKConfig()
    depth_config = DepthConfig()
    calibration_config = CalibrationConfig()
    measurement_config = MeasurementConfig()
    backend = SyntheticCameraBackend() if args.demo else None
    camera = SynexensCamera(sdk_config, backend=backend)
    preprocessor = DepthPreprocessor(depth_config)
    calibrator = PlatformCalibrator(calibration_config)
    estimator = make_estimator(measurement_config)

    try:
        info = camera.open()
        camera.start()
        print(f"Connected: {info.model} serial={info.serial_number or 'unknown'}")
        scale_model = load_depth_scale_model()
        if scale_model is not None and not _MIN_REASONABLE_DEPTH_SCALE <= scale_model.scale <= _MAX_REASONABLE_DEPTH_SCALE:
            print(
                f"Ignoring implausible saved depth scale {scale_model.scale:.6f}. "
                "Press c and enter the measured height again."
            )
            scale_model = None
        if args.depth_reference_mm is not None:
            scale_model = calibrate_depth_scale(camera, args.depth_reference_mm)
            if not args.calibrate and not args.measure:
                return 0
        # Command-line automation may reuse a saved model. Interactive use must
        # start uncalibrated so the operator explicitly captures today's empty platform.
        platform_model = load_platform_model()
        if args.calibrate:
            print("Capturing empty-platform frames for verified ROI proposal...")
            capture = capture_platform_proposal(camera, calibrator, calibration_config, preprocessor)
            print(proposal_summary(capture.proposal))
            try:
                platform_model, calibrated_scale_model = calibrate_platform(
                    camera,
                    calibrator,
                    calibration_config,
                    preprocessor,
                    args.platform_height_mm,
                    capture,
                )
            except ValueError as exc:
                print(f"Platform calibration rejected: {exc}")
                return 2
            if calibrated_scale_model is not None:
                scale_model = calibrated_scale_model
            if not args.measure and not args.detect:
                return 0
        if args.detect:
            return print_detection(camera, estimator, preprocessor, scale_model, platform_model)
        if args.measure:
            if platform_model is None:
                print("Measurement skipped: platform_plane.json is not available. Run --calibrate first.")
                return 2
            frame = _next_frame(camera)
            _, cloud = clean_frame(frame, preprocessor, scale_model)
            result, dynamic_model = measure_with_dynamic_baseline(estimator, calibrator, cloud, platform_model)
            print(
                f"Dynamic baseline: {dynamic_model.camera_height_mm:.2f} mm "
                f"(delta {dynamic_model.camera_height_mm - platform_model.camera_height_mm:+.2f} mm)"
            )
            print(
                "Dynamic ROI plane residuals (mm): "
                f"p05={dynamic_model.residual_p05_mm:.2f}, "
                f"p50={dynamic_model.residual_p50_mm:.2f}, "
                f"p95={dynamic_model.residual_p95_mm:.2f}, "
                f"std={dynamic_model.residual_std_mm:.2f}"
            )
            print_result(result)
            return 0 if result.success else 1
        if args.headless:
            frame = _next_frame(camera)
            cleaned, _ = clean_frame(frame, preprocessor, scale_model)
            valid_count = int(np.isfinite(cleaned.depth_mm).sum())
            print(f"Depth frame: {cleaned.width}x{cleaned.height}, valid points={valid_count}")
            return 0
        interactive_loop(
            camera, calibrator, estimator, calibration_config, preprocessor, scale_model, None
        )
        return 0
    except CameraError as exc:
        print(f"Camera error: {exc}")
        return 2
    finally:
        camera.stop()
        camera.close()


if __name__ == "__main__":
    raise SystemExit(main())

"""Application composition root for the CS20 volume scanner.

This file intentionally coordinates components only.  Synexens SDK calls stay
inside ``camera/synexens_camera.py``; all measurement code works with clean
millimetre arrays and calibration models.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Iterable

import numpy as np

from calibration.calibration_store import (
    load_depth_scale_model,
    load_platform_model,
    save_depth_scale_model,
    save_platform_model,
)
from calibration.depth_scale_calibrator import DepthScaleCalibrator, DepthScaleModel
from calibration.platform_calibrator import PlatformCalibrator, PlatformModel
from camera.frame_types import DepthFrame
from camera.synexens_camera import CameraError, SynexensCamera, SyntheticCameraBackend
from config import CalibrationConfig, DepthConfig, MeasurementConfig, SDKConfig
from measurement.box_detector import BoxDetector
from measurement.rectangle_fitter import RectangleFitter
from measurement.top_surface import TopSurfaceExtractor
from measurement.volume_estimator import BoxVolumeEstimator, VolumeResult
from processing.depth_filter import DepthPreprocessor
from processing.pointcloud_utils import depth_to_pointcloud
from visualization.viewer import Viewer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CS20 box-volume scanner")
    parser.add_argument("--demo", action="store_true", help="use a deterministic synthetic CS20 frame source")
    parser.add_argument("--headless", action="store_true", help="do not open an OpenCV window")
    parser.add_argument("--calibrate", action="store_true", help="save an empty-platform model and exit")
    parser.add_argument("--detect", action="store_true", help="print only box point count and ROI, then exit")
    parser.add_argument("--measure", action="store_true", help="perform one measurement and exit")
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
    cleaned_frame = frame.with_depth(cleaned_depth)
    return cleaned_frame, depth_to_pointcloud(cleaned_depth, frame.intrinsics)


def collect_pointclouds(
    camera: SynexensCamera,
    count: int,
    preprocessor: DepthPreprocessor,
    scale_model: DepthScaleModel | None,
) -> tuple[list[np.ndarray], str]:
    points: list[np.ndarray] = []
    resolution = ""
    while len(points) < count:
        frame = camera.read()
        if frame is None:
            continue
        cleaned, cloud = clean_frame(frame, preprocessor, scale_model)
        resolution = cleaned.resolution_name
        points.append(cloud)
    return points, resolution


def make_estimator(config: MeasurementConfig) -> BoxVolumeEstimator:
    """Wire the final detector → top surface → rectangle pipeline."""

    detector = BoxDetector(config)
    top_surface = TopSurfaceExtractor(config)
    rectangle = RectangleFitter(config)
    return BoxVolumeEstimator(config, detector, top_surface, rectangle)


def print_result(result: VolumeResult) -> None:
    data = asdict(result)
    print(data)


def print_detection(camera: SynexensCamera, estimator: BoxVolumeEstimator, preprocessor: DepthPreprocessor,
                    scale_model: DepthScaleModel | None, platform_model: PlatformModel | None) -> int:
    if platform_model is None:
        print("Detection skipped: platform_plane.json is not available. Run --calibrate first.")
        return 2
    frame = _next_frame(camera)
    _, cloud = clean_frame(frame, preprocessor, scale_model)
    detection = estimator.detect_box(cloud, platform_model)
    print({"found": detection.found, "box_point_count": detection.point_count, "roi": detection.roi})
    return 0 if detection.found else 1


def calibrate_platform(
    camera: SynexensCamera,
    calibrator: PlatformCalibrator,
    config: CalibrationConfig,
    preprocessor: DepthPreprocessor,
    scale_model: DepthScaleModel | None,
) -> PlatformModel:
    preprocessor.reset()
    points, resolution = collect_pointclouds(camera, config.frame_count, preprocessor, scale_model)
    model = calibrator.calibrate(points, resolution=resolution)
    path = save_platform_model(model)
    print(f"Platform plane saved to {path}; camera_height_mm={model.camera_height_mm:.2f}")
    return model


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
    show_ir = False
    result: VolumeResult | None = None
    try:
        while True:
            cleaned, cloud, _ = run_once_measurement(camera, estimator, preprocessor, scale_model, None)
            key = viewer.show_depth(
                cleaned, depth_mm=cleaned.depth_mm, show_ir=show_ir, result=result, platform_model=platform_model
            )
            if key == "q":
                return
            if key == "i":
                show_ir = not show_ir
            elif key == "b":
                if platform_model is None:
                    print("No platform model. Press c with an empty platform first.")
                else:
                    detection = estimator.detect_box(cloud, platform_model)
                    print({"found": detection.found, "box_point_count": detection.point_count, "roi": detection.roi})
            elif key == "c":
                print("Calibrating platform: keep the platform empty...")
                platform_model = calibrate_platform(
                    camera, calibrator, calibration_config, preprocessor, scale_model
                )
                result = None
            elif key == "d":
                if platform_model is None:
                    print("No platform model. Press c with an empty platform first.")
                else:
                    result = estimator.measure(cloud, platform_model)
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
        if args.depth_reference_mm is not None:
            scale_model = calibrate_depth_scale(camera, args.depth_reference_mm)
            if not args.calibrate and not args.measure:
                return 0
        platform_model = load_platform_model()
        if args.calibrate:
            platform_model = calibrate_platform(
                camera, calibrator, calibration_config, preprocessor, scale_model
            )
            if not args.measure and not args.detect:
                return 0
        if args.detect:
            return print_detection(camera, estimator, preprocessor, scale_model, platform_model)
        if args.measure:
            _, _, result = run_once_measurement(
                camera, estimator, preprocessor, scale_model, platform_model
            )
            if result is None:
                print("Measurement skipped: platform_plane.json is not available. Run --calibrate first.")
                return 2
            print_result(result)
            return 0 if result.success else 1
        if args.headless:
            frame = _next_frame(camera)
            cleaned, _ = clean_frame(frame, preprocessor, scale_model)
            valid_count = int(np.isfinite(cleaned.depth_mm).sum())
            print(f"Depth frame: {cleaned.width}x{cleaned.height}, valid points={valid_count}")
            return 0
        interactive_loop(
            camera, calibrator, estimator, calibration_config, preprocessor, scale_model, platform_model
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

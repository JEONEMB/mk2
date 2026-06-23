"""Configuration for the volume-scanner pipeline.

All distances in this project are millimetres unless a name states otherwise.
"""

from dataclasses import dataclass
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SDK_ROOT = Path(os.environ.get("SYNEXENS_SDK_ROOT", r"C:\SDK\SynexensSDK_4.2.5.0_windows"))


@dataclass(frozen=True)
class SDKConfig:
    """Settings consumed only by :mod:`camera.synexens_camera`."""

    # Set SYNEXENS_SDK_ROOT to override this location on another PC.
    sdk_root: Path | None = DEFAULT_SDK_ROOT
    # The supplied C# demo contains a complete native runtime in this folder.
    # ``sdk_root`` stays first, so a normal SDK installation takes precedence.
    fallback_sdk_roots: tuple[Path, ...] = (
        PROJECT_ROOT / "vendor" / "SynexensSDK",
        PROJECT_ROOT / "vendor" / "SynexensCSharpDemo" / "SDKTest" / "bin" / "x64" / "Release",
        Path.home() / "Downloads" / "SynexensSDK_4.2.5.0 CSharpDemo" / "SDKTest" / "bin" / "x64" / "Release",
    )
    device_index: int = 0
    # Synexens' CS20 C# sample configures depth/IR at 320x240.
    default_resolution: str = "320x240"
    enable_ir: bool = True
    frame_timeout_ms: int = 1_000


@dataclass(frozen=True)
class DepthConfig:
    min_valid_mm: float = 100.0
    max_valid_mm: float = 4_000.0
    ir_saturation_threshold: int = 250
    ir_saturation_dilate_px: int = 3
    spatial_median_kernel: int = 3
    temporal_window: int = 15


@dataclass(frozen=True)
class CalibrationConfig:
    frame_count: int = 60
    roi_ratio: float = 0.60
    ransac_iterations: int = 300
    ransac_threshold_mm: float = 4.0
    min_inliers: int = 1_000
    max_sample_points: int = 30_000
    dynamic_floor_update_threshold_mm: float = 8.0


@dataclass(frozen=True)
class MeasurementConfig:
    min_box_height_mm: float = 15.0
    min_box_points: int = 250
    box_close_kernel: int = 5
    top_percentile: float = 80.0
    top_band_mm: float = 5.0
    top_plane_threshold_mm: float = 4.0
    max_top_plane_residual_std_mm: float = 4.0
    min_top_inlier_ratio: float = 0.65
    min_top_points: int = 80
    rectangle_trim_percentile_low: float = 3.0
    rectangle_trim_percentile_high: float = 97.0


@dataclass(frozen=True)
class PathConfig:
    calibration_dir: Path = Path("data") / "calibration"
    sample_dir: Path = Path("data") / "samples"
    platform_plane_path: Path = calibration_dir / "platform_plane.json"
    depth_scale_path: Path = calibration_dir / "depth_scale.json"

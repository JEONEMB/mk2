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
    # Raw CS20 IR is uint16. Select only the brightest per-frame tail instead
    # of applying an 8-bit threshold to the entire image.
    ir_saturation_percentile: float = 99.8
    ir_min_dynamic_range: float = 128.0
    ir_max_mask_ratio: float = 0.05
    ir_saturation_dilate_px: int = 3
    spatial_median_kernel: int = 3
    temporal_window: int = 15


@dataclass(frozen=True)
class CalibrationConfig:
    frame_count: int = 60
    roi_ratio: float = 0.60
    ransac_iterations: int = 300
    ransac_threshold_mm: float = 4.0
    # The precise far-point seed uses ``ransac_threshold_mm``.  Full-frame
    # support is allowed a little more range noise, then the ROI residual gate
    # below decides whether the surface is truly usable for measurement.
    full_frame_plane_threshold_mm: float = 10.0
    min_inliers: int = 1_000
    max_sample_points: int = 30_000
    # Initial platform candidates are the farthest 40% of metric point-cloud
    # distances. Nearer boxes and foreground obstacles are rejected first,
    # while enough platform points remain after vertical walls are excluded.
    platform_farthest_percentile: float = 60.0
    # A platform candidate is one raw 8-connected inlier component.  Small
    # holes may be bridged only inside that component while finding its
    # rectangle; they never connect two candidate components.
    platform_plane_candidate_count: int = 3
    platform_roi_hole_close_px: int = 5
    min_platform_roi_coverage: float = 0.90
    platform_max_tilt_deg: float = 25.0
    min_platform_component_pixels: int = 1_000
    min_platform_roi_area_ratio: float = 0.10
    min_calibration_valid_depth_ratio: float = 0.15
    min_measurement_roi_side_px: int = 64
    max_calibration_residual_std_mm: float = 3.0
    max_calibration_residual_abs_p95_mm: float = 8.0
    # Do not measure directly on the automatically found platform boundary.
    # At 320x240 this removes a 16 px guard band from each ROI edge.
    measurement_roi_inset_px: int = 16
    # At measurement time refit a live plane from the low-height part of the
    # ROI.  The guards reject a genuinely different scene rather than small
    # camera movement or range drift.
    dynamic_baseline_max_shift_mm: float = 60.0
    dynamic_baseline_min_points: int = 500
    dynamic_plane_iterations: int = 200
    dynamic_plane_threshold_mm: float = 6.0
    dynamic_plane_max_tilt_deg: float = 8.0
    dynamic_floor_low_percentile: float = 5.0
    dynamic_floor_high_percentile: float = 60.0


@dataclass(frozen=True)
class MeasurementConfig:
    # A 70 mm box should be well above this while low platform waviness and
    # residual depth noise stay out of the candidate mask.
    min_box_height_mm: float = 40.0
    # A 130 x 80 mm box at 700 mm in 320x240 commonly occupies a few hundred
    # pixels; 80 retains it while the rectangular platform ROI rejects clutter.
    min_box_points: int = 80
    box_close_kernel: int = 3
    top_percentile: float = 80.0
    top_band_mm: float = 5.0
    top_plane_threshold_mm: float = 4.0
    max_top_plane_residual_std_mm: float = 4.0
    min_top_inlier_ratio: float = 0.65
    min_top_points: int = 300
    rectangle_trim_percentile_low: float = 3.0
    rectangle_trim_percentile_high: float = 97.0


@dataclass(frozen=True)
class PathConfig:
    calibration_dir: Path = Path("data") / "calibration"
    sample_dir: Path = Path("data") / "samples"
    platform_plane_path: Path = calibration_dir / "platform_plane.json"
    platform_mask_path: Path = calibration_dir / "platform_plane_mask.npy"
    depth_scale_path: Path = calibration_dir / "depth_scale.json"

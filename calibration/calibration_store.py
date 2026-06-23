"""Versioned JSON persistence for platform and depth-scale calibration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from calibration.depth_scale_calibrator import DepthScaleModel
from calibration.platform_calibrator import PlatformModel
from config import PathConfig


def save_json(path: Path | str, data: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path: Path | str) -> dict[str, Any] | None:
    target = Path(path)
    if not target.exists():
        return None
    data = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"calibration JSON must contain an object: {target}")
    return data


def save_platform_model(
    platform_model: PlatformModel | Path | str, path: Path | str | PlatformModel | None = None
) -> Path:
    """Save a platform model (also accepts legacy ``path, model`` ordering)."""

    if isinstance(platform_model, PlatformModel):
        model = platform_model
        target = Path(path) if path is not None else PathConfig().platform_plane_path  # type: ignore[arg-type]
    else:
        if not isinstance(path, PlatformModel):
            raise TypeError("legacy save_platform_model call requires (path, PlatformModel)")
        target, model = Path(platform_model), path
    save_json(target, model.to_dict())
    return target


def load_platform_model(path: Path | str | None = None) -> PlatformModel | None:
    target = Path(path) if path is not None else PathConfig().platform_plane_path
    data = load_json(target)
    return None if data is None else PlatformModel.from_dict(data)


def save_depth_scale_model(
    scale_model: DepthScaleModel | Path | str, path: Path | str | DepthScaleModel | None = None
) -> Path:
    """Save a depth model (also accepts legacy ``path, model`` ordering)."""

    if isinstance(scale_model, DepthScaleModel):
        model = scale_model
        target = Path(path) if path is not None else PathConfig().depth_scale_path  # type: ignore[arg-type]
    else:
        if not isinstance(path, DepthScaleModel):
            raise TypeError("legacy save_depth_scale_model call requires (path, DepthScaleModel)")
        target, model = Path(scale_model), path
    save_json(target, model.to_dict())
    return target


def load_depth_scale_model(path: Path | str | None = None) -> DepthScaleModel | None:
    target = Path(path) if path is not None else PathConfig().depth_scale_path
    data = load_json(target)
    return None if data is None else DepthScaleModel.from_dict(data)

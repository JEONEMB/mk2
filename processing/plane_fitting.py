"""Robust plane estimation independent of the camera SDK."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlaneFit:
    normal: np.ndarray
    d: float
    inlier_mask: np.ndarray
    residual_std_mm: float


def _normalise_plane(normal: np.ndarray, d: float) -> tuple[np.ndarray, float]:
    normal = np.asarray(normal, dtype=np.float64)
    magnitude = np.linalg.norm(normal)
    if magnitude == 0:
        raise ValueError("a plane normal cannot be zero")
    normal, d = normal / magnitude, float(d) / magnitude
    # Face the camera. Objects above the platform then have positive height.
    if normal[2] > 0:
        normal, d = -normal, -d
    return normal, d


def _finite_points(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return values[np.isfinite(values).all(axis=1)]


def fit_plane_svd(points: np.ndarray) -> PlaneFit:
    values = _finite_points(points)
    if len(values) < 3:
        raise ValueError("at least three finite points are required for a plane")
    center = values.mean(axis=0)
    _, _, vh = np.linalg.svd(values - center, full_matrices=False)
    normal, d = _normalise_plane(vh[-1], -np.dot(vh[-1], center))
    residuals = point_to_plane_signed_distance(values, normal, d)
    return PlaneFit(normal, d, np.ones(len(values), dtype=bool), float(np.std(residuals)))


def point_to_plane_signed_distance(points: np.ndarray, normal: np.ndarray, d: float) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    return values @ np.asarray(normal, dtype=np.float64) + float(d)


def fit_plane_ransac(
    points: np.ndarray,
    iterations: int = 300,
    threshold_mm: float = 4.0,
    min_inliers: int = 1_000,
    rng: np.random.Generator | None = None,
) -> PlaneFit:
    values = _finite_points(points)
    if len(values) < 3:
        raise ValueError("at least three finite points are required for a plane")
    generator = rng or np.random.default_rng(0)
    best_mask: np.ndarray | None = None
    best_score = (-1, np.inf)
    for _ in range(iterations):
        sample = values[generator.choice(len(values), 3, replace=False)]
        cross = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        if np.linalg.norm(cross) < 1e-9:
            continue
        normal, d = _normalise_plane(cross, -np.dot(cross, sample[0]))
        residuals = np.abs(point_to_plane_signed_distance(values, normal, d))
        mask = residuals <= threshold_mm
        score = (int(mask.sum()), float(residuals[mask].mean()) if mask.any() else np.inf)
        if score[0] > best_score[0] or (score[0] == best_score[0] and score[1] < best_score[1]):
            best_mask, best_score = mask, score
    if best_mask is None or int(best_mask.sum()) < min(3, min_inliers):
        raise ValueError("RANSAC could not find a sufficiently large plane")
    refined = fit_plane_svd(values[best_mask])
    all_mask = np.abs(point_to_plane_signed_distance(values, refined.normal, refined.d)) <= threshold_mm
    if int(all_mask.sum()) < min_inliers:
        raise ValueError(f"plane has {int(all_mask.sum())} inliers; expected at least {min_inliers}")
    final = fit_plane_svd(values[all_mask])
    return PlaneFit(final.normal, final.d, all_mask, final.residual_std_mm)


def refine_plane_with_inliers(points: np.ndarray, normal: np.ndarray, d: float, threshold_mm: float) -> PlaneFit:
    values = _finite_points(points)
    mask = np.abs(point_to_plane_signed_distance(values, normal, d)) <= threshold_mm
    if int(mask.sum()) < 3:
        raise ValueError("not enough inliers to refine a plane")
    fit = fit_plane_svd(values[mask])
    return PlaneFit(fit.normal, fit.d, mask, fit.residual_std_mm)


def update_plane_d_only(points: np.ndarray, base_normal: np.ndarray, base_d: float, threshold_mm: float) -> float:
    """Update only the offset, preserving the calibrated platform orientation."""

    values = _finite_points(points)
    distances = point_to_plane_signed_distance(values, base_normal, base_d)
    inliers = values[np.abs(distances) <= threshold_mm]
    if len(inliers) < 3:
        return float(base_d)
    # For n·x + d = 0, each inlier gives d = -n·x.
    return float(np.median(-(inliers @ np.asarray(base_normal))))

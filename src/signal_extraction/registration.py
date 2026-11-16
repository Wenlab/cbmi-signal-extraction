"""Red/calcium channel registration matching the improved GPU workflow."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np


DEFAULT_MAX_TRANSLATION_DEVIATION_PX = 3.0


class RegistrationCalibrationError(RuntimeError):
    """Raised when sampled ECC cannot produce a robust calibration matrix."""

    def __init__(self, message: str, diagnostics: Dict[str, Any]):
        super().__init__(message)
        self.diagnostics = diagnostics


def _normalize_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32).copy()
    image -= image.min()
    image = np.where(image > 3000, 102, image)
    maximum = image.max()
    if maximum > 0:
        image /= maximum
    return image


def estimate_registration_matrix(
    red_mip: np.ndarray,
    green_mip: np.ndarray,
    flip_horizontal: bool = True,
    max_translation_deviation_px: float = DEFAULT_MAX_TRANSLATION_DEVIATION_PX,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Estimate the guarded full-Z-MIP ECC matrix used by GPU improved."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for ECC registration. Install opencv-python, "
            "provide registration_matrix, or set register_calcium=False."
        ) from exc

    red_mip = np.asarray(red_mip, dtype=np.float32)
    green_mip = np.asarray(green_mip, dtype=np.float32)
    if red_mip.ndim != 2 or green_mip.ndim != 2:
        raise ValueError("ECC inputs must both be 2-D Z-MIP images")
    if red_mip.shape != green_mip.shape:
        raise ValueError(
            "red and calcium registration MIPs differ in shape: %r vs %r"
            % (red_mip.shape, green_mip.shape)
        )

    identity = np.eye(2, 3, dtype=np.float32)

    diagnostics = {
        "reason": None,
        "correlation": None,
        "rotation_deviation_deg": None,
        "translation_deviation_px": None,
        "estimated_matrix": None,
        "initial_matrix": identity.tolist(),
    }
    green_for_ecc = np.fliplr(green_mip) if flip_horizontal else green_mip
    red_norm = _normalize_image(red_mip)
    green_norm = _normalize_image(green_for_ecc)
    if (
        not np.isfinite(red_norm).all()
        or not np.isfinite(green_norm).all()
        or float(np.ptp(red_norm)) == 0.0
        or float(np.ptp(green_norm)) == 0.0
    ):
        diagnostics["reason"] = "registration MIP is empty or non-finite"
        raise RegistrationCalibrationError(diagnostics["reason"], diagnostics)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        5000,
        1e-7,
    )
    try:
        correlation, estimated = cv2.findTransformECC(
            red_norm,
            green_norm,
            identity.copy(),
            cv2.MOTION_EUCLIDEAN,
            criteria,
            None,
            5,
        )
    except cv2.error as exc:
        diagnostics["reason"] = "ECC did not converge: %s" % (
            str(exc).splitlines()[0],
        )
        raise RegistrationCalibrationError(diagnostics["reason"], diagnostics) from exc

    estimated = np.asarray(estimated, dtype=np.float32)
    diagnostics["correlation"] = float(correlation)
    diagnostics["estimated_matrix"] = estimated.tolist()
    if not np.isfinite(correlation) or not np.isfinite(estimated).all():
        diagnostics["reason"] = "ECC returned non-finite correlation or matrix"
        raise RegistrationCalibrationError(diagnostics["reason"], diagnostics)

    estimated_angle = np.degrees(np.arctan2(estimated[1, 0], estimated[0, 0]))
    angle_delta = abs((float(estimated_angle) + 180.0) % 360.0 - 180.0)
    translation_delta = float(np.linalg.norm(estimated[:, 2]))
    diagnostics["rotation_deviation_deg"] = angle_delta
    diagnostics["translation_deviation_px"] = translation_delta

    if translation_delta > float(max_translation_deviation_px):
        diagnostics["reason"] = "translation deviation %.3f px" % translation_delta
        raise RegistrationCalibrationError(diagnostics["reason"], diagnostics)

    return estimated, diagnostics


def median_euclidean_matrix(matrices: Sequence[np.ndarray]) -> np.ndarray:
    """Build a robust Euclidean matrix exactly as ``ecc_alignment_only.py``."""
    if not matrices:
        raise ValueError("at least one matrix is required")
    matrices_np = np.asarray(matrices, dtype=np.float32)
    if matrices_np.ndim != 3 or matrices_np.shape[1:] != (2, 3):
        raise ValueError("matrices must have shape (K,2,3)")
    angles = np.arctan2(matrices_np[:, 1, 0], matrices_np[:, 0, 0])
    angle = float(np.median(angles))
    tx = float(np.median(matrices_np[:, 0, 2]))
    ty = float(np.median(matrices_np[:, 1, 2]))
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.asarray(
        [[cosine, -sine, tx], [sine, cosine, ty]], dtype=np.float32
    )


def estimate_sampled_registration_matrix(
    reference_loaders: Sequence[Callable[[], np.ndarray]],
    calcium_loaders: Sequence[Callable[[], np.ndarray]],
    *,
    frame_ids: Optional[Sequence[int]] = None,
    num_samples: int = 10,
    seed: int = 72,
    min_correlation: float = 0.1,
    min_successful_samples: int = 3,
    max_translation_deviation_px: float = DEFAULT_MAX_TRANSLATION_DEVIATION_PX,
    flip_horizontal: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Estimate a robust matrix from random full-volume Z-MIP ECC samples.

    This mirrors ``scripts/ecc_alignment_only.py``:

    1. sample up to ten paired volumes without replacement using seed 72;
    2. run Euclidean ECC on each pair from the identity matrix;
    3. keep estimates whose ECC correlation passes the threshold;
    4. form a preliminary median rotation/translation matrix;
    5. reject matrices inconsistent with that median;
    6. take the median again over the consistent estimates.

    Failed samples are discarded instead of being replaced by a fallback
    matrix.
    """
    if len(reference_loaders) != len(calcium_loaders):
        raise ValueError("reference and calcium loader counts differ")
    total_volumes = len(reference_loaders)
    if total_volumes == 0:
        raise ValueError("no paired volumes are available for registration")
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")
    if min_successful_samples < 1:
        raise ValueError("min_successful_samples must be at least 1")
    if not -1.0 <= float(min_correlation) <= 1.0:
        raise ValueError("min_correlation must be between -1 and 1")
    if frame_ids is None:
        frame_ids = list(range(total_volumes))
    if len(frame_ids) != total_volumes:
        raise ValueError("frame_ids length differs from paired volume count")

    sample_count = min(int(num_samples), total_volumes)
    rng = np.random.default_rng(int(seed))
    sampled_positions = sorted(
        int(index)
        for index in rng.choice(total_volumes, size=sample_count, replace=False)
    )
    sampled_frame_ids = [int(frame_ids[index]) for index in sampled_positions]
    sample_diagnostics = []
    candidates = []

    for position, frame_id in zip(sampled_positions, sampled_frame_ids):
        sample = {
            "position": int(position),
            "frame_id": int(frame_id),
            "correlation": None,
            "estimated_matrix": None,
            "candidate": False,
            "consistent": False,
            "error": None,
        }
        try:
            reference_volume = np.asarray(reference_loaders[position]())
            calcium_volume = np.asarray(calcium_loaders[position]())
            if reference_volume.ndim != 3 or calcium_volume.ndim != 3:
                raise ValueError("sampled volumes must both have shape (Z,Y,X)")
            reference_mip = reference_volume.max(axis=0)
            calcium_mip = calcium_volume.max(axis=0)
            _, ecc_diagnostics = estimate_registration_matrix(
                reference_mip,
                calcium_mip,
                flip_horizontal=flip_horizontal,
                max_translation_deviation_px=1.0e9,
            )
            correlation = ecc_diagnostics["correlation"]
            estimated = ecc_diagnostics["estimated_matrix"]
            sample["correlation"] = correlation
            sample["estimated_matrix"] = estimated
            sample["error"] = ecc_diagnostics["reason"]
            if (
                correlation is not None
                and np.isfinite(correlation)
                and float(correlation) >= float(min_correlation)
                and estimated is not None
            ):
                sample["candidate"] = True
                candidates.append(
                    (len(sample_diagnostics), np.asarray(estimated, dtype=np.float32))
                )
        except RegistrationCalibrationError as exc:
            sample["error"] = "%s: %s" % (type(exc).__name__, exc)
        sample_diagnostics.append(sample)

    metadata: Dict[str, Any] = {
        "strategy": "sample_median",
        "num_available_volumes": int(total_volumes),
        "num_requested_samples": int(num_samples),
        "num_sampled_volumes": int(sample_count),
        "sampled_positions": sampled_positions,
        "sampled_frame_ids": sampled_frame_ids,
        "seed": int(seed),
        "min_correlation": float(min_correlation),
        "min_successful_samples": int(min_successful_samples),
        "max_translation_deviation_px": float(max_translation_deviation_px),
        "initial_matrix": np.eye(2, 3, dtype=np.float32).tolist(),
        "samples": sample_diagnostics,
        "num_ecc_candidates": int(len(candidates)),
        "num_consistent_candidates": 0,
        "preliminary_matrix": None,
        "matrix": None,
    }

    consistent = []
    if len(candidates) >= min_successful_samples:
        preliminary = median_euclidean_matrix([matrix for _, matrix in candidates])
        metadata["preliminary_matrix"] = preliminary.tolist()
        preliminary_angle = math.degrees(
            math.atan2(float(preliminary[1, 0]), float(preliminary[0, 0]))
        )
        for sample_index, matrix in candidates:
            angle = math.degrees(
                math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
            )
            angle_delta = abs(
                (angle - preliminary_angle + 180.0) % 360.0 - 180.0
            )
            translation_delta = float(
                np.linalg.norm(matrix[:, 2] - preliminary[:, 2])
            )
            sample_diagnostics[sample_index]["rotation_deviation_deg"] = angle_delta
            sample_diagnostics[sample_index]["translation_deviation_px"] = (
                translation_delta
            )
            if translation_delta <= float(max_translation_deviation_px):
                sample_diagnostics[sample_index]["consistent"] = True
                consistent.append(matrix)

    metadata["num_consistent_candidates"] = int(len(consistent))
    if len(consistent) < min_successful_samples:
        raise RegistrationCalibrationError(
            "sampled ECC retained %d consistent matrices; at least %d are required"
            % (len(consistent), min_successful_samples),
            metadata,
        )

    calibrated = median_euclidean_matrix(consistent)
    metadata["matrix"] = calibrated.tolist()
    return calibrated, metadata


def transform_reference_to_calcium_coords(
    neuron_pt_tuple: np.ndarray,
    matrix: np.ndarray,
    calcium_width: int,
    flip_horizontal: bool = True,
) -> np.ndarray:
    """Map reference coordinates into the original calcium image space."""
    coords = np.asarray(neuron_pt_tuple).copy()
    matrix = np.asarray(matrix, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("neuron coordinates must have shape (N,F), F>=2")
    if matrix.shape != (2, 3):
        raise ValueError("registration matrix must have shape (2,3)")

    x = coords[:, 0].copy()
    y = coords[:, 1].copy()
    x_flipped = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
    y_flipped = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]
    coords[:, 0] = (
        calcium_width - 1 - x_flipped if flip_horizontal else x_flipped
    )
    coords[:, 1] = y_flipped
    return coords

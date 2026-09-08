"""End-to-end signal extraction from neuron bounding boxes and raw images."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import warnings

import h5py
import numpy as np
from tqdm.auto import tqdm

from .coordinates import CoordinateFrame, load_coordinate_frames
from .core import extract_volume_signals, resolve_device
from .images import VolumeFrame, VolumeSeries, open_volume_series
from .registration import (
    DEFAULT_MAX_TRANSLATION_DEVIATION_PX,
    estimate_registration_matrix,
    estimate_sampled_registration_matrix,
    transform_reference_to_calcium_coords,
)


@dataclass
class SignalExtractionResult:
    """Complete extraction result, aligned as ``(channel, neuron, time)``."""

    activity: np.ndarray
    neuron_ids: np.ndarray
    frame_ids: np.ndarray
    reference_z_start: np.ndarray
    reference_z_count: np.ndarray
    registration_matrix: Optional[np.ndarray]
    registration_diagnostics: Optional[Dict[str, Any]]
    device: str
    parameters: Dict[str, Any]


def _match_image_frames(
    coordinates: Sequence[CoordinateFrame],
    images: VolumeSeries,
    alignment: str,
    channel_name: str,
) -> List[VolumeFrame]:
    alignment = alignment.lower()
    if alignment not in ("auto", "id", "position"):
        raise ValueError("frame_alignment must be auto, id or position")

    by_id = images.by_id
    ids_match = all(frame.frame_id in by_id for frame in coordinates)
    if alignment == "id" or (alignment == "auto" and ids_match):
        missing = [frame.frame_id for frame in coordinates if frame.frame_id not in by_id]
        if missing:
            raise ValueError(
                "%s images are missing inference frame IDs: %s"
                % (channel_name, missing[:10])
            )
        return [by_id[frame.frame_id] for frame in coordinates]

    if len(images) != len(coordinates):
        raise ValueError(
            "%s image count (%d) differs from inference frame count (%d); "
            "frame-ID matching also failed"
            % (channel_name, len(images), len(coordinates))
        )
    if alignment == "auto":
        warnings.warn(
            "%s image IDs do not match inference IDs; aligning by sorted position"
            % channel_name,
            RuntimeWarning,
            stacklevel=2,
        )
    return list(images.frames)


def _load_registration_matrix(
    registration_matrix: Optional[Union[str, Path, np.ndarray]],
) -> Optional[np.ndarray]:
    if registration_matrix is None:
        return None
    if isinstance(registration_matrix, (str, Path)):
        matrix = np.load(registration_matrix)
    else:
        matrix = np.asarray(registration_matrix)
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise ValueError("registration_matrix must be finite shape (2,3)")
    return matrix


def _write_result(path: Union[str, Path], result: SignalExtractionResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("activity", data=result.activity, compression="gzip")
        handle.create_dataset("neuron_ids", data=result.neuron_ids)
        handle.create_dataset("frame_ids", data=result.frame_ids)
        handle.create_dataset(
            "reference_z_start", data=result.reference_z_start, compression="gzip"
        )
        handle.create_dataset(
            "reference_z_count", data=result.reference_z_count, compression="gzip"
        )
        if result.registration_matrix is not None:
            handle.create_dataset(
                "registration_matrix", data=result.registration_matrix
            )
        handle.attrs["channels"] = np.asarray(
            ["reference", "calcium"][:result.activity.shape[0]],
            dtype=h5py.string_dtype("utf-8"),
        )
        handle.attrs["device"] = result.device
        handle.attrs["algorithm"] = "gpu-improved-compatible-torch"
        handle.attrs["parameters_json"] = json.dumps(result.parameters)
        if result.registration_diagnostics is not None:
            handle.attrs["registration_diagnostics_json"] = json.dumps(
                result.registration_diagnostics
            )


def extract_signals(
    inference_result: Union[str, Path, np.ndarray],
    reference_images: Union[str, Path],
    calcium_images: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    *,
    device: str = "auto",
    percentile_reference: float = 0.01,
    percentile_calcium: float = 0.01,
    area_ratio: float = 1.0,
    background_threshold: float = 102.0,
    bounding_box_format: str = "auto",
    input_format: str = "auto",
    axis_order: str = "auto",
    frames_per_volume: int = 20,
    tiff_layout: str = "auto",
    z_range: Optional[Tuple[int, int]] = (0, 18),
    frame_alignment: str = "auto",
    register_calcium: bool = True,
    registration_strategy: str = "sample_median",
    registration_num_samples: int = 10,
    registration_seed: int = 72,
    registration_min_correlation: float = 0.1,
    registration_min_successful_samples: int = 3,
    registration_frame: int = 0,
    registration_matrix: Optional[Union[str, Path, np.ndarray]] = None,
    max_translation_deviation_px: float = DEFAULT_MAX_TRANSLATION_DEVIATION_PX,
    flip_calcium_horizontal: bool = True,
    show_progress: bool = True,
) -> SignalExtractionResult:
    """Extract signals from neuron bounding boxes and raw images.

    Both ``device='cpu'`` and ``device='cuda'`` execute the same Torch
    algorithm as GPU improved. Images are loaded one volume at a time.

    Parameters
    ----------
    inference_result:
        A dynamics HDF5, coordinate NPY ``(T,N,F)``, aggregate realtime HDF5,
        or directory containing realtime ``volume_<id>.h5`` files.
    bounding_box_format:
        ``auto``, ``dynamics``, ``npy`` or ``realtime-results``. Automatic
        detection uses whether the source is a directory and its suffix/HDF5
        structure.
    reference_images, calcium_images:
        NPY/TIFF file or directory. See the README for supported layouts.
    output_path:
        Optional output HDF5. The primary dataset is ``activity`` with shape
        ``(channels, neurons, time)``.
    register_calcium:
        Estimate ECC unless ``registration_matrix`` is supplied. Setting this
        false uses an identity affine matrix; horizontal mirroring is governed
        separately by ``flip_calcium_horizontal``.
    registration_strategy:
        ``"sample_median"`` (default) randomly samples paired volumes and
        robustly combines their full-Z-MIP ECC matrices, matching
        ``scripts/ecc_alignment_only.py``. ``"single"`` retains the older
        guarded ECC estimate from ``registration_frame``.
    """
    resolved_device = resolve_device(device)
    reference_series = open_volume_series(
        reference_images,
        input_format=input_format,
        axis_order=axis_order,
        frames_per_volume=frames_per_volume,
        tiff_layout=tiff_layout,
        z_range=z_range,
    )
    coordinate_frames = load_coordinate_frames(
        inference_result,
        source_format=bounding_box_format,
        expected_frame_ids=(
            [frame.frame_id for frame in reference_series.frames]
            if isinstance(inference_result, (str, Path))
            and Path(inference_result).is_dir()
            else None
        ),
    )
    reference_frames = _match_image_frames(
        coordinate_frames, reference_series, frame_alignment, "reference"
    )

    calcium_frames: Optional[List[VolumeFrame]] = None
    if calcium_images is not None:
        calcium_series = open_volume_series(
            calcium_images,
            input_format=input_format,
            axis_order=axis_order,
            frames_per_volume=frames_per_volume,
            tiff_layout=tiff_layout,
            z_range=z_range,
        )
        calcium_frames = _match_image_frames(
            coordinate_frames, calcium_series, frame_alignment, "calcium"
        )

    all_ids = sorted(
        {
            int(neuron_id)
            for frame in coordinate_frames
            for neuron_id in frame.neuron_ids
        }
    )
    if not all_ids:
        raise ValueError("bounding-box source contains no non-negative neuron IDs")
    neuron_ids = np.asarray(all_ids, dtype=np.int64)
    id_to_row = {neuron_id: row for row, neuron_id in enumerate(all_ids)}
    time_count = len(coordinate_frames)
    channel_count = 2 if calcium_frames is not None else 1
    activity = np.full(
        (channel_count, len(all_ids), time_count), np.nan, dtype=np.float32
    )
    reference_z_start = np.full(
        (len(all_ids), time_count), -1, dtype=np.int32
    )
    reference_z_count = np.zeros(
        (len(all_ids), time_count), dtype=np.int32
    )

    accepted_matrix: Optional[np.ndarray] = None
    registration_diagnostics: Optional[Dict[str, Any]] = None
    if calcium_frames is not None:
        registration_strategy = registration_strategy.lower()
        if registration_strategy not in ("sample_median", "single"):
            raise ValueError(
                "registration_strategy must be sample_median or single"
            )
        supplied_matrix = _load_registration_matrix(registration_matrix)
        if supplied_matrix is not None:
            accepted_matrix = supplied_matrix
            registration_diagnostics = {
                "reason": "user-supplied registration matrix",
                "source": "supplied",
                "strategy": "supplied",
            }
        elif register_calcium:
            if registration_strategy == "sample_median":
                accepted_matrix, registration_diagnostics = (
                    estimate_sampled_registration_matrix(
                        [frame.load for frame in reference_frames],
                        [frame.load for frame in calcium_frames],
                        frame_ids=[frame.frame_id for frame in coordinate_frames],
                        num_samples=registration_num_samples,
                        seed=registration_seed,
                        min_correlation=registration_min_correlation,
                        min_successful_samples=(
                            registration_min_successful_samples
                        ),
                        max_translation_deviation_px=(
                            max_translation_deviation_px
                        ),
                        flip_horizontal=flip_calcium_horizontal,
                    )
                )
                registration_diagnostics["source"] = "sampled_ecc_median"
                registration_diagnostics["reason"] = None
            else:
                if registration_frame < 0 or registration_frame >= time_count:
                    raise IndexError(
                        "registration_frame %d is outside [0,%d)"
                        % (registration_frame, time_count)
                    )
                red_registration_volume = reference_frames[
                    registration_frame
                ].load()
                calcium_registration_volume = calcium_frames[
                    registration_frame
                ].load()
                accepted_matrix, registration_diagnostics = (
                    estimate_registration_matrix(
                        red_registration_volume.max(axis=0),
                        calcium_registration_volume.max(axis=0),
                        flip_horizontal=flip_calcium_horizontal,
                        max_translation_deviation_px=max_translation_deviation_px,
                    )
                )
                registration_diagnostics["strategy"] = "single"
                registration_diagnostics["source"] = "ecc"
                registration_diagnostics["registration_frame_position"] = int(
                    registration_frame
                )
                registration_diagnostics["registration_frame_id"] = int(
                    coordinate_frames[registration_frame].frame_id
                )
        else:
            accepted_matrix = np.eye(2, 3, dtype=np.float32)
            registration_diagnostics = {
                "reason": "ECC registration disabled; identity affine used",
                "source": "identity",
                "strategy": "disabled",
            }

    iterator = zip(coordinate_frames, reference_frames)
    iterator = tqdm(
        iterator,
        total=time_count,
        desc="Signal extraction (%s)" % resolved_device,
        disable=not show_progress,
    )
    for time_index, (coordinate_frame, reference_frame) in enumerate(iterator):
        coords = coordinate_frame.coordinates
        rows = np.asarray(
            [id_to_row[int(neuron_id)] for neuron_id in coordinate_frame.neuron_ids],
            dtype=np.int64,
        )
        reference_volume = reference_frame.load()
        reference_result = extract_volume_signals(
            reference_volume,
            coords,
            percentile=percentile_reference,
            area_ratio=area_ratio,
            background_threshold=background_threshold,
            device=resolved_device,
        )
        activity[0, rows, time_index] = reference_result.intensities
        reference_z_start[rows, time_index] = reference_result.z_start
        reference_z_count[rows, time_index] = reference_result.z_count

        if calcium_frames is not None:
            calcium_volume = calcium_frames[time_index].load()
            calcium_coords = transform_reference_to_calcium_coords(
                coords,
                accepted_matrix,
                calcium_width=calcium_volume.shape[2],
                flip_horizontal=flip_calcium_horizontal,
            )
            calcium_result = extract_volume_signals(
                calcium_volume,
                calcium_coords,
                percentile=percentile_calcium,
                area_ratio=area_ratio,
                background_threshold=background_threshold,
                device=resolved_device,
                forced_z_start=reference_result.z_start,
            )
            activity[1, rows, time_index] = calcium_result.intensities

    result = SignalExtractionResult(
        activity=activity,
        neuron_ids=neuron_ids,
        frame_ids=np.asarray(
            [frame.frame_id for frame in coordinate_frames], dtype=np.int64
        ),
        reference_z_start=reference_z_start,
        reference_z_count=reference_z_count,
        registration_matrix=accepted_matrix,
        registration_diagnostics=registration_diagnostics,
        device=str(resolved_device),
        parameters={
            "percentile_reference": float(percentile_reference),
            "percentile_calcium": float(percentile_calcium),
            "area_ratio": float(area_ratio),
            "background_threshold": float(background_threshold),
            "bounding_box_format": str(bounding_box_format),
            "axis_order": str(axis_order),
            "frames_per_volume": int(frames_per_volume),
            "tiff_layout": str(tiff_layout),
            "z_range": None if z_range is None else [int(z_range[0]), int(z_range[1])],
            "frame_alignment": str(frame_alignment),
            "register_calcium": bool(register_calcium),
            "registration_strategy": str(registration_strategy),
            "registration_num_samples": int(registration_num_samples),
            "registration_seed": int(registration_seed),
            "registration_min_correlation": float(
                registration_min_correlation
            ),
            "registration_min_successful_samples": int(
                registration_min_successful_samples
            ),
            "registration_frame": int(registration_frame),
            "max_translation_deviation_px": float(
                max_translation_deviation_px
            ),
            "flip_calcium_horizontal": bool(flip_calcium_horizontal),
        },
    )
    if output_path is not None:
        _write_result(output_path, result)
    return result

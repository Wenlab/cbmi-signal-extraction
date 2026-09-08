"""Bounding-box readers for dynamics HDF5, NumPy and realtime results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import List, Optional, Union

import h5py
import numpy as np


_COORDINATE_DATASETS = (
    "d_neuron_pt_tuple_matched_raw_vol",
    "neuron_pt_tuple",
)
_VOLUME_NUMBER = re.compile(r"volume_(\d+)(?:-(\d+))?", re.IGNORECASE)
_TRAILING_NUMBER = re.compile(r"(\d+)(?!.*\d)")
_REALTIME_FILE = re.compile(r"volume_(\d+)\.h5", re.IGNORECASE)
_REALTIME_GROUP = re.compile(r"volume_(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class CoordinateFrame:
    """Coordinates and stable neuron IDs for one inference time point."""

    frame_id: int
    name: str
    coordinates: np.ndarray
    neuron_ids: np.ndarray


def _number_from_name(name: str, fallback: int) -> int:
    volume_match = _VOLUME_NUMBER.search(name)
    if volume_match:
        if volume_match.group(2) is not None:
            return -int(volume_match.group(2))
        return int(volume_match.group(1))
    trailing_match = _TRAILING_NUMBER.search(name)
    return int(trailing_match.group(1)) if trailing_match else int(fallback)


def _natural_frame_key(name: str):
    number = _number_from_name(name, -1)
    return (number < 0, number, name)


def _natural_name_key(name: str):
    """Sort names with every embedded decimal field compared numerically."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", name)
    ]


def _normalise_frame(
    coordinates: np.ndarray,
    neuron_ids: np.ndarray,
    frame_id: int,
    name: str,
) -> CoordinateFrame:
    coordinates = np.asarray(coordinates)
    neuron_ids = np.asarray(neuron_ids)
    if coordinates.ndim != 2 or coordinates.shape[1] < 5:
        raise ValueError(
            "%s coordinates must have shape (N,F), F>=5; got %r"
            % (name, coordinates.shape)
        )
    if neuron_ids.shape != (coordinates.shape[0],):
        raise ValueError(
            "%s neuron IDs have shape %r, expected (%d,)"
            % (name, neuron_ids.shape, coordinates.shape[0])
        )
    neuron_ids = neuron_ids.astype(np.int64, copy=False)
    valid_ids = neuron_ids >= 0
    coordinates = coordinates[valid_ids]
    neuron_ids = neuron_ids[valid_ids]
    if np.unique(neuron_ids).size != neuron_ids.size:
        raise ValueError("%s contains duplicate non-negative neuron IDs" % name)
    return CoordinateFrame(
        frame_id=int(frame_id),
        name=str(name),
        coordinates=np.asarray(coordinates, dtype=np.float32),
        neuron_ids=neuron_ids,
    )


def _frames_from_array(coordinates: np.ndarray) -> List[CoordinateFrame]:
    coordinates = np.asarray(coordinates)
    if coordinates.ndim == 2:
        coordinates = coordinates[np.newaxis, ...]
    if coordinates.ndim != 3:
        raise ValueError(
            "coordinate NPY/array must have shape (T,N,F) or (N,F), got %r"
            % (coordinates.shape,)
        )
    return [
        _normalise_frame(
            coordinates[index],
            np.arange(coordinates.shape[1], dtype=np.int64),
            index,
            "volume_%08d" % index,
        )
        for index in range(coordinates.shape[0])
    ]


def _read_group_frame(
    group: h5py.Group,
    frame_id: int,
    name: str,
    require_neuron_ids: bool = False,
) -> CoordinateFrame:
    dataset_name = next(
        (candidate for candidate in _COORDINATE_DATASETS if candidate in group),
        None,
    )
    if dataset_name is None:
        raise KeyError("no supported coordinate dataset in %s" % group.name)
    coordinates = group[dataset_name][()]
    if "neuron_pred_ids" in group:
        neuron_ids = group["neuron_pred_ids"][()]
        if not np.issubdtype(np.asarray(neuron_ids).dtype, np.integer):
            raise TypeError("%s neuron_pred_ids must be integers" % group.name)
    elif require_neuron_ids:
        raise KeyError("%s does not contain neuron_pred_ids" % group.name)
    else:
        neuron_ids = np.arange(coordinates.shape[0], dtype=np.int64)
    return _normalise_frame(coordinates, neuron_ids, frame_id, name)


def _validate_unique_frame_ids(frames: List[CoordinateFrame]) -> List[CoordinateFrame]:
    if not frames:
        raise ValueError("coordinate source contains no bounding-box frames")
    frame_ids = [frame.frame_id for frame in frames]
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError("bounding-box source contains duplicate frame IDs")
    return frames


def _frames_from_h5(
    path: Path, source_format: str = "auto"
) -> List[CoordinateFrame]:
    frames: List[CoordinateFrame] = []
    with h5py.File(path, "r") as handle:
        is_realtime = "real-time_analysis" in handle
        if source_format == "dynamics" and is_realtime:
            raise ValueError("%s is realtime results, not dynamics HDF5" % path)
        if source_format == "realtime-results" and not is_realtime:
            raise ValueError("%s has no /real-time_analysis group" % path)

        if is_realtime:
            root = handle["real-time_analysis"]
            names = sorted(root.keys(), key=_natural_frame_key)
            for position, name in enumerate(names):
                item = root[name]
                if not isinstance(item, h5py.Group):
                    continue
                if not any(key in item for key in _COORDINATE_DATASETS):
                    continue
                frame_id = _number_from_name(name, position)
                # Realtime files may contain warm-up keys such as volume_0-1.
                # They do not correspond to acquired image frame IDs.
                if frame_id < 0:
                    continue
                frames.append(
                    _read_group_frame(
                        item,
                        frame_id,
                        name,
                        require_neuron_ids=True,
                    )
                )
        else:
            # Also accept a root-level dense (T,N,F) dataset.
            dense_name = next(
                (
                    candidate
                    for candidate in _COORDINATE_DATASETS
                    if candidate in handle
                    and isinstance(handle[candidate], h5py.Dataset)
                ),
                None,
            )
            if dense_name is not None:
                dense = handle[dense_name][()]
                if dense.ndim == 3:
                    return _frames_from_array(dense)

            names = sorted(
                (
                    name
                    for name, item in handle.items()
                    if isinstance(item, h5py.Group)
                    and not re.search(r"(?i)warm|metadata|outcomes", name)
                    and any(key in item for key in _COORDINATE_DATASETS)
                ),
                key=_natural_name_key,
            )
            for position, name in enumerate(names):
                item = handle[name]
                # Match roi_source.py: plain numeric dynamics group names are
                # source volume IDs; descriptive acquisition names are mapped
                # to their position after natural sorting. Their final number
                # can restart for every ImgStk and is therefore not globally
                # unique.
                frame_id = int(name) if name.isdigit() else position
                frames.append(_read_group_frame(item, frame_id, name))

    if not frames:
        raise ValueError(
            "No bounding-box frames found in %s. Expected dynamics "
            "per-volume d_neuron_pt_tuple_matched_raw_vol/neuron_pt_tuple or "
            "realtime /real-time_analysis/.../neuron_pt_tuple." % path
        )
    return _validate_unique_frame_ids(frames)


def _find_realtime_group(
    handle: h5py.File, frame_id: int
) -> Optional[h5py.Group]:
    analysis = handle.get("real-time_analysis")
    if not isinstance(analysis, h5py.Group):
        return None
    expected_name = "volume_%08d" % frame_id
    expected = analysis.get(expected_name)
    if isinstance(expected, h5py.Group):
        return expected
    for name, item in analysis.items():
        match = _REALTIME_GROUP.fullmatch(name)
        if (
            match is not None
            and int(match.group(1)) == frame_id
            and isinstance(item, h5py.Group)
        ):
            return item
    return None


def _read_realtime_result_file(
    path: Optional[Path], frame_id: int
) -> Optional[CoordinateFrame]:
    """Read one ``volume_<id>.h5`` exactly like realtime ``roi_source.py``."""
    if path is None:
        return None
    with h5py.File(path, "r") as handle:
        group = _find_realtime_group(handle, frame_id)
        if group is None:
            return None
        if "neuron_pt_tuple" not in group or "neuron_pred_ids" not in group:
            return None
        frame = _read_group_frame(
            group,
            frame_id,
            "volume_%08d" % frame_id,
            require_neuron_ids=True,
        )
    if frame.coordinates.shape[1] < 6:
        raise ValueError(
            "Realtime points in %s must have shape (N,F>=6), got %r"
            % (path, frame.coordinates.shape)
        )
    return frame


def _frames_from_realtime_directory(
    path: Path, expected_frame_ids: Optional[List[int]] = None
) -> List[CoordinateFrame]:
    files_by_id = {}
    for file_path in path.iterdir():
        if not file_path.is_file():
            continue
        match = _REALTIME_FILE.fullmatch(file_path.name)
        if match is None:
            continue
        frame_id = int(match.group(1))
        if frame_id in files_by_id:
            raise ValueError(
                "realtime directory contains duplicate files for volume %d"
                % frame_id
            )
        files_by_id[frame_id] = file_path
    if not files_by_id:
        raise ValueError("No volume_*.h5 files found in %s" % path)

    frame_cache = {}

    def read_frame(frame_id: int) -> Optional[CoordinateFrame]:
        if frame_id not in frame_cache:
            frame_cache[frame_id] = _read_realtime_result_file(
                files_by_id.get(frame_id), frame_id
            )
        return frame_cache[frame_id]

    baseline = None
    for frame_id in sorted(files_by_id):
        candidate = read_frame(frame_id)
        if candidate is not None and candidate.neuron_ids.size > 0:
            baseline = candidate
            break
    if baseline is None:
        raise ValueError("No realtime neuron point data found in %s" % path)

    requested_ids = (
        sorted(files_by_id)
        if expected_frame_ids is None
        else [int(frame_id) for frame_id in expected_frame_ids]
    )
    if len(set(requested_ids)) != len(requested_ids):
        raise ValueError("expected realtime frame IDs contain duplicates")
    loaded = [(frame_id, read_frame(frame_id)) for frame_id in requested_ids]
    valid_frames = [baseline] + [
        frame for _, frame in loaded if frame is not None
    ]
    feature_counts = {frame.coordinates.shape[1] for frame in valid_frames}
    if len(feature_counts) != 1:
        raise ValueError(
            "Realtime point feature counts are inconsistent: %r"
            % sorted(feature_counts)
        )
    feature_count = feature_counts.pop()

    # roi_source.py treats a missing/incomplete realtime result as an empty
    # time point. Preserve it so image and signal time axes cannot shift.
    frames = []
    for frame_id, frame in loaded:
        if frame is None:
            frame = _normalise_frame(
                np.empty((0, feature_count), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                frame_id,
                "volume_%08d" % frame_id,
            )
        frames.append(frame)
    return _validate_unique_frame_ids(frames)


def load_coordinate_frames(
    source: Union[str, Path, np.ndarray],
    source_format: str = "auto",
    expected_frame_ids: Optional[List[int]] = None,
) -> List[CoordinateFrame]:
    """Load bounding boxes from dynamics HDF5, NPY or realtime results.

    ``realtime-results`` accepts either one aggregate realtime HDF5 or the
    ``volume_<id>.h5`` directory layout consumed by ``roi_source.py``.
    """
    selected_format = source_format.lower().replace("_", "-")
    if selected_format == "realtime":
        selected_format = "realtime-results"
    valid_formats = ("auto", "dynamics", "npy", "realtime-results")
    if selected_format not in valid_formats:
        raise ValueError("source_format must be one of %s" % (valid_formats,))
    if isinstance(source, np.ndarray):
        if selected_format not in ("auto", "npy"):
            raise ValueError("an in-memory array requires source_format='npy'")
        return _frames_from_array(source)
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError("bounding-box source does not exist: %s" % path)
    if path.is_dir():
        if selected_format not in ("auto", "realtime-results"):
            raise ValueError(
                "a bounding-box directory requires realtime-results format"
            )
        return _frames_from_realtime_directory(path, expected_frame_ids)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        if selected_format not in ("auto", "npy"):
            raise ValueError("NPY input requires source_format='npy'")
        return _frames_from_array(np.load(path, mmap_mode="r"))
    if suffix in (".h5", ".hdf5"):
        if selected_format == "npy":
            raise ValueError("HDF5 input cannot use source_format='npy'")
        return _frames_from_h5(path, selected_format)
    raise ValueError(
        "bounding-box source must be dynamics/realtime HDF5, NPY, or a "
        "realtime volume_*.h5 directory: %s" % path
    )

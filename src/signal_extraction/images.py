"""Streaming NPY and TIFF volume readers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import tifffile


_LAST_NUMBER = re.compile(r"(\d+)(?!.*\d)")


def _number_from_path(path: Path, fallback: int) -> int:
    match = _LAST_NUMBER.search(path.stem)
    return int(match.group(1)) if match else int(fallback)


def _normalise_3d_volume(array: np.ndarray, axis_order: str) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 3:
        raise ValueError("expected a 3-D volume, got shape %r" % (array.shape,))
    order = axis_order.lower()
    if order == "tzyx":
        order = "zyx"
    elif order == "tyxz":
        order = "yxz"
    if order == "auto":
        first_is_z = array.shape[0] <= 64 and array.shape[0] < min(array.shape[1:])
        last_is_z = array.shape[2] <= 64 and array.shape[2] < min(array.shape[:2])
        order = "zyx" if first_is_z or not last_is_z else "yxz"
    if order == "zyx":
        volume = array
    elif order == "yxz":
        volume = array.transpose(2, 0, 1)
    else:
        raise ValueError("3-D axis_order must be auto, zyx or yxz")
    return np.ascontiguousarray(volume)


def _normalise_4d_series(array: np.ndarray, axis_order: str) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 4:
        raise ValueError("expected a 4-D series, got shape %r" % (array.shape,))
    order = axis_order.lower()
    if order == "zyx":
        order = "tzyx"
    elif order == "yxz":
        order = "tyxz"
    if order == "auto":
        first_is_z = array.shape[1] <= 64 and array.shape[1] < min(array.shape[2:])
        last_is_z = array.shape[3] <= 64 and array.shape[3] < min(array.shape[1:3])
        order = "tzyx" if first_is_z or not last_is_z else "tyxz"
    if order == "tzyx":
        series = array
    elif order == "tyxz":
        series = array.transpose(0, 3, 1, 2)
    else:
        raise ValueError("4-D axis_order must be auto, tzyx or tyxz")
    return series


def _slice_z(volume: np.ndarray, z_range: Optional[Tuple[int, int]]) -> np.ndarray:
    if z_range is None:
        return volume
    start, stop = z_range
    return volume[int(start):int(stop)]


@dataclass(frozen=True)
class VolumeFrame:
    frame_id: int
    name: str
    loader: Callable[[], np.ndarray]

    def load(self) -> np.ndarray:
        volume = np.asarray(self.loader())
        if volume.ndim != 3:
            raise ValueError(
                "%s did not load as (Z,Y,X); got %r" % (self.name, volume.shape)
            )
        return np.ascontiguousarray(volume)


@dataclass
class VolumeSeries:
    frames: List[VolumeFrame]

    def __post_init__(self):
        if not self.frames:
            raise ValueError("image series is empty")
        ids = [frame.frame_id for frame in self.frames]
        if len(set(ids)) != len(ids):
            raise ValueError("image series contains duplicate frame IDs")

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def by_id(self) -> Dict[int, VolumeFrame]:
        return {frame.frame_id: frame for frame in self.frames}


def _npy_series(
    path: Path, axis_order: str, z_range: Optional[Tuple[int, int]]
) -> VolumeSeries:
    if path.is_dir():
        files = sorted(path.glob("*.npy"), key=lambda p: (_number_from_path(p, -1), p.name))
        frames = []
        for position, file_path in enumerate(files):
            frame_id = _number_from_path(file_path, position)

            def load_one(p=file_path):
                return _slice_z(
                    _normalise_3d_volume(np.load(p), axis_order), z_range
                )

            frames.append(VolumeFrame(frame_id, file_path.stem, load_one))
        return VolumeSeries(frames)

    array = np.load(path, mmap_mode="r")
    if array.ndim == 3:
        return VolumeSeries([
            VolumeFrame(
                0,
                path.stem,
                lambda: _slice_z(_normalise_3d_volume(array, axis_order), z_range),
            )
        ])
    if array.ndim == 4:
        series = _normalise_4d_series(array, axis_order)
        return VolumeSeries([
            VolumeFrame(
                index,
                "%s_%08d" % (path.stem, index),
                lambda i=index: _slice_z(np.asarray(series[i]), z_range),
            )
            for index in range(series.shape[0])
        ])
    raise ValueError("NPY must have shape (Z,Y,X) or (T,Z,Y,X)")


def _tiff_directory_as_slices(
    files: Sequence[Path],
    frames_per_volume: int,
    z_range: Optional[Tuple[int, int]],
) -> VolumeSeries:
    if frames_per_volume <= 0:
        raise ValueError("frames_per_volume must be positive")
    numbered = [(_number_from_path(path, position), path) for position, path in enumerate(files)]
    numbered.sort(key=lambda item: (item[0], item[1].name))
    grouped: Dict[int, List[Tuple[int, Path]]] = {}
    for frame_number, path in numbered:
        grouped.setdefault(frame_number // frames_per_volume, []).append((frame_number, path))

    frames = []
    for volume_id in sorted(grouped):
        members = grouped[volume_id]
        if len(members) != frames_per_volume:
            raise ValueError(
                "TIFF volume %d has %d slices; expected %d"
                % (volume_id, len(members), frames_per_volume)
            )

        def load_slices(items=tuple(members)):
            volume = np.stack([tifffile.imread(path) for _, path in items], axis=0)
            return _slice_z(volume, z_range)

        frames.append(VolumeFrame(volume_id, "volume_%08d" % volume_id, load_slices))
    return VolumeSeries(frames)


def _tiff_series(
    path: Path,
    axis_order: str,
    frames_per_volume: int,
    tiff_layout: str,
    z_range: Optional[Tuple[int, int]],
) -> VolumeSeries:
    layout = tiff_layout.lower()
    if path.is_dir():
        files = sorted(
            list(path.glob("*.tif")) + list(path.glob("*.tiff")),
            key=lambda p: (_number_from_path(p, -1), p.name),
        )
        if not files:
            raise ValueError("no TIFF files found in %s" % path)
        if layout == "auto":
            layout = "slices" if tifffile.imread(files[0]).ndim == 2 else "volumes"
        if layout == "slices":
            return _tiff_directory_as_slices(files, frames_per_volume, z_range)
        if layout != "volumes":
            raise ValueError("tiff_layout must be auto, slices or volumes")

        frames = []
        for position, file_path in enumerate(files):
            frame_id = _number_from_path(file_path, position)

            def load_volume(p=file_path):
                return _slice_z(
                    _normalise_3d_volume(tifffile.imread(p), axis_order), z_range
                )

            frames.append(VolumeFrame(frame_id, file_path.stem, load_volume))
        return VolumeSeries(frames)

    # memmap avoids loading an uncompressed time series eagerly. Compressed
    # TIFF cannot be memory-mapped, so fall back to tifffile.imread.
    try:
        array = tifffile.memmap(path)
    except (ValueError, OSError):
        array = tifffile.imread(path)
    if array.ndim == 3:
        return VolumeSeries([
            VolumeFrame(
                0,
                path.stem,
                lambda: _slice_z(_normalise_3d_volume(array, axis_order), z_range),
            )
        ])
    if array.ndim == 4:
        series = _normalise_4d_series(array, axis_order)
        return VolumeSeries([
            VolumeFrame(
                index,
                "%s_%08d" % (path.stem, index),
                lambda i=index: _slice_z(np.asarray(series[i]), z_range),
            )
            for index in range(series.shape[0])
        ])
    raise ValueError("TIFF must contain a 3-D volume or a 4-D time series")


def open_volume_series(
    source: Union[str, Path],
    input_format: str = "auto",
    axis_order: str = "auto",
    frames_per_volume: int = 20,
    tiff_layout: str = "auto",
    z_range: Optional[Tuple[int, int]] = None,
) -> VolumeSeries:
    """Open a directory/file of NPY or TIFF images without loading all volumes."""
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError("image source does not exist: %s" % path)
    selected_format = input_format.lower()
    if selected_format == "auto":
        if path.is_file():
            suffix = path.suffix.lower()
            selected_format = "npy" if suffix == ".npy" else "tiff" if suffix in (".tif", ".tiff") else ""
        else:
            has_npy = any(path.glob("*.npy"))
            has_tiff = any(path.glob("*.tif")) or any(path.glob("*.tiff"))
            if has_npy and has_tiff:
                raise ValueError(
                    "directory contains both NPY and TIFF; set input_format explicitly"
                )
            selected_format = "npy" if has_npy else "tiff" if has_tiff else ""
    if selected_format == "npy":
        return _npy_series(path, axis_order, z_range)
    if selected_format == "tiff":
        return _tiff_series(
            path, axis_order, frames_per_volume, tiff_layout, z_range
        )
    raise ValueError("could not detect NPY/TIFF image format for %s" % path)

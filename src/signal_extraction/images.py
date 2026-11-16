"""Streaming NPY, HDF5 and TIFF volume readers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import warnings

import h5py
import numpy as np
import tifffile


_LAST_NUMBER = re.compile(r"(\d+)(?!.*\d)")

# 网络共享（SMB）上的偶发 OSError（如 Errno 22）通常是瞬时的，重试可恢复。
_LOAD_ATTEMPTS = 5
_LOAD_RETRY_DELAY_SECONDS = 2.0


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


def _tiff_axis_order(axis_order: str, axes: str) -> str:
    if axis_order.lower() == "auto" and axes.lower() in (
        "zyx", "yxz", "tzyx", "tyxz"
    ):
        return axes.lower()
    return axis_order


@dataclass(frozen=True)
class VolumeFrame:
    frame_id: int
    name: str
    loader: Callable[[], np.ndarray]

    def load(self) -> np.ndarray:
        for attempt in range(_LOAD_ATTEMPTS):
            try:
                volume = np.asarray(self.loader())
                break
            except OSError:
                if attempt == _LOAD_ATTEMPTS - 1:
                    raise
                warnings.warn(
                    "transient error loading %s; retrying (%d/%d)"
                    % (self.name, attempt + 1, _LOAD_ATTEMPTS - 1),
                    RuntimeWarning,
                    stacklevel=2,
                )
                time.sleep(_LOAD_RETRY_DELAY_SECONDS)
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
    path: Path,
    axis_order: str,
    z_range: Optional[Tuple[int, int]],
    max_volumes: Optional[int] = None,
) -> VolumeSeries:
    if max_volumes is not None and max_volumes <= 0:
        raise ValueError("max_volumes must be a positive integer")
    if path.is_dir():
        files = sorted(path.glob("*.npy"), key=lambda p: (_number_from_path(p, -1), p.name))
        frames = []
        for position, file_path in enumerate(files):
            frame_id = _number_from_path(file_path, position)

            def load_one(p=file_path):
                return _slice_z(
                    _normalise_3d_volume(
                        np.load(p, mmap_mode="r", allow_pickle=False), axis_order
                    ),
                    z_range,
                )

            frames.append(VolumeFrame(frame_id, file_path.stem, load_one))
        if max_volumes is not None:
            frames = frames[:max_volumes]
        return VolumeSeries(frames)

    array = np.load(path, mmap_mode="r", allow_pickle=False)
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
        volume_count = series.shape[0]
        if max_volumes is not None:
            volume_count = min(volume_count, max_volumes)
        return VolumeSeries([
            VolumeFrame(
                index,
                "%s_%08d" % (path.stem, index),
                lambda i=index: _slice_z(np.asarray(series[i]), z_range),
            )
            for index in range(volume_count)
        ])
    raise ValueError("NPY must have shape (Z,Y,X) or (T,Z,Y,X)")


def _h5_series(
    path: Path,
    axis_order: str,
    z_range: Optional[Tuple[int, int]],
    max_volumes: Optional[int] = None,
) -> VolumeSeries:
    """Read the exporter's fixed /volumes (T,Z,Y,X) layout one T at a time."""
    if not path.is_file():
        raise ValueError("HDF5 image source must be a file, not a directory")
    if axis_order.lower() not in ("auto", "zyx", "tzyx"):
        raise ValueError("HDF5 /volumes has fixed TZYX axes; use auto or tzyx")
    if max_volumes is not None and max_volumes <= 0:
        raise ValueError("max_volumes must be a positive integer")

    with h5py.File(path, "r") as handle:
        if "volumes" not in handle:
            raise ValueError("HDF5 image file must contain /volumes: %s" % path)
        images = handle["volumes"]
        if not isinstance(images, h5py.Dataset):
            raise ValueError("HDF5 /volumes must be a dataset")
        if images.ndim != 4 or any(size == 0 for size in images.shape):
            raise ValueError("HDF5 /volumes must have non-empty shape (T,Z,Y,X)")
        if not np.issubdtype(images.dtype, np.number) or np.issubdtype(
            images.dtype, np.complexfloating
        ):
            raise ValueError("HDF5 /volumes must contain real numeric images")
        axes = images.attrs.get("axes", "TZYX")
        if isinstance(axes, bytes):
            axes = axes.decode("ascii")
        if axes != "TZYX":
            raise ValueError("HDF5 /volumes axes must be TZYX, got %r" % axes)

        filters = images.id.get_create_plist()
        filter_ids = [
            filters.get_filter(index)[0] for index in range(filters.get_nfilters())
        ]
        if any(not h5py.h5z.filter_avail(key) for key in filter_ids):
            try:
                import hdf5plugin  # noqa: F401 -- registers Zstd HDF5 filters.
            except ImportError as error:
                raise ImportError(
                    "HDF5 image compression requires hdf5plugin; run pixi install"
                ) from error
            unavailable = [
                key for key in filter_ids if not h5py.h5z.filter_avail(key)
            ]
            if unavailable:
                raise ValueError("Unavailable HDF5 image filters: %s" % unavailable)
        volume_count = images.shape[0]

    if max_volumes is not None:
        volume_count = min(volume_count, max_volumes)
    z_slice = slice(None) if z_range is None else slice(*z_range)

    def load_volume(index: int) -> np.ndarray:
        # Reopen per load so no VolumeFrame retains a live HDF5 handle.
        with h5py.File(path, "r") as handle:
            return handle["volumes"][index, z_slice, :, :]

    return VolumeSeries([
        VolumeFrame(
            index,
            "%s_%08d" % (path.stem, index),
            lambda i=index: load_volume(i),
        )
        for index in range(volume_count)
    ])


def _tiff_directory_as_slices(
    files: Sequence[Path],
    frames_per_volume: int,
    z_range: Optional[Tuple[int, int]],
    max_volumes: Optional[int] = None,
) -> VolumeSeries:
    if frames_per_volume <= 0:
        raise ValueError("frames_per_volume must be positive")
    if max_volumes is not None and max_volumes <= 0:
        raise ValueError("max_volumes must be a positive integer")
    numbered = [(_number_from_path(path, position), path) for position, path in enumerate(files)]
    numbered.sort(key=lambda item: (item[0], item[1].name))
    grouped: Dict[int, List[Tuple[int, Path]]] = {}
    for frame_number, path in numbered:
        grouped.setdefault(frame_number // frames_per_volume, []).append((frame_number, path))

    frames = []
    for volume_id in sorted(grouped):
        if max_volumes is not None and volume_id >= max_volumes:
            break
        members = grouped[volume_id]
        if len(members) != frames_per_volume:
            warnings.warn(
                "TIFF volume %d has %d slices; expected %d — skipping "
                "incomplete volume" % (volume_id, len(members), frames_per_volume),
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        def load_slices(items=tuple(members)):
            volume = np.stack([tifffile.imread(path) for _, path in items], axis=0)
            return _slice_z(volume, z_range)

        frames.append(VolumeFrame(volume_id, "volume_%08d" % volume_id, load_slices))
    return VolumeSeries(frames)


def _compressed_tiff_series(
    path: Path,
    axis_order: str,
    z_range: Optional[Tuple[int, int]],
    max_volumes: Optional[int],
) -> VolumeSeries:
    """Index TIFF pages once, then decode only the requested complete volume."""
    with tifffile.TiffFile(path) as handle:
        series = handle.series[0]
        shape = series.shape
        if len(shape) not in (3, 4):
            raise ValueError("TIFF must contain a 3-D volume or a 4-D time series")
        if any(size == 0 for size in shape):
            raise ValueError("TIFF image dimensions must be non-empty")
        volume_shape = shape[-3:]
        volume_count = shape[0] if len(shape) == 4 else 1
        pages_per_volume = volume_shape[0]
        if (
            len(series.pages) != volume_count * pages_per_volume
            or series.pages[0].shape != volume_shape[1:]
        ):
            raise ValueError(
                "Compressed TIFF must store one grayscale 2-D plane per page"
            )
        dtype = series.dtype
        order = _tiff_axis_order(axis_order, series.axes)
        if max_volumes is not None:
            volume_count = min(volume_count, max_volumes)
        # Store only metadata, never pixel arrays or an open TiffFile.
        page_offsets = tuple(
            series.pages[index].offset
            for index in range(volume_count * pages_per_volume)
        )

    def load_volume(index: int) -> np.ndarray:
        volume = np.empty(volume_shape, dtype=dtype)
        first_page = index * pages_per_volume
        with tifffile.TiffFile(path) as handle:
            keyframe = handle.pages[0]
            for z_index in range(pages_per_volume):
                page_index = first_page + z_index
                page = (
                    keyframe
                    if page_index == 0
                    else tifffile.TiffFrame(
                        handle,
                        page_index,
                        offset=page_offsets[page_index],
                        keyframe=keyframe,
                    )
                )
                page.asarray(out=volume[z_index], maxworkers=1)
        return _slice_z(_normalise_3d_volume(volume, order), z_range)

    return VolumeSeries([
        VolumeFrame(
            index,
            "%s_%08d" % (path.stem, index),
            lambda i=index: load_volume(i),
        )
        for index in range(volume_count)
    ])


def _tiff_series(
    path: Path,
    axis_order: str,
    frames_per_volume: int,
    tiff_layout: str,
    z_range: Optional[Tuple[int, int]],
    max_volumes: Optional[int] = None,
) -> VolumeSeries:
    if max_volumes is not None and max_volumes <= 0:
        raise ValueError("max_volumes must be a positive integer")
    layout = tiff_layout.lower()
    if path.is_dir():
        files = sorted(
            list(path.glob("*.tif")) + list(path.glob("*.tiff")),
            key=lambda p: (_number_from_path(p, -1), p.name),
        )
        if not files:
            raise ValueError("no TIFF files found in %s" % path)
        if layout == "auto":
            with tifffile.TiffFile(files[0]) as handle:
                layout = "slices" if len(handle.series[0].shape) == 2 else "volumes"
        if layout == "slices":
            return _tiff_directory_as_slices(
                files, frames_per_volume, z_range, max_volumes
            )
        if layout != "volumes":
            raise ValueError("tiff_layout must be auto, slices or volumes")

        frames = []
        for position, file_path in enumerate(files):
            frame_id = _number_from_path(file_path, position)

            def load_volume(p=file_path):
                with tifffile.TiffFile(p) as handle:
                    series = handle.series[0]
                    order = _tiff_axis_order(axis_order, series.axes)
                    return _slice_z(
                        _normalise_3d_volume(series.asarray(), order), z_range
                    )

            frames.append(VolumeFrame(frame_id, file_path.stem, load_volume))
            if max_volumes is not None and len(frames) >= max_volumes:
                break
        return VolumeSeries(frames)

    # Keep memory mapping for compatible uncompressed TIFFs. Compressed stacks
    # must be decoded per volume, not eagerly via tifffile.imread(path).
    with tifffile.TiffFile(path) as handle:
        order = _tiff_axis_order(axis_order, handle.series[0].axes)
    try:
        array = tifffile.memmap(path, mode="r")
    except (ValueError, OSError):
        return _compressed_tiff_series(path, order, z_range, max_volumes)
    if array.ndim == 3:
        return VolumeSeries([
            VolumeFrame(
                0,
                path.stem,
                lambda: _slice_z(_normalise_3d_volume(array, order), z_range),
            )
        ])
    if array.ndim == 4:
        series = _normalise_4d_series(array, order)
        volume_count = series.shape[0]
        if max_volumes is not None:
            volume_count = min(volume_count, max_volumes)
        return VolumeSeries([
            VolumeFrame(
                index,
                "%s_%08d" % (path.stem, index),
                lambda i=index: _slice_z(np.asarray(series[i]), z_range),
            )
            for index in range(volume_count)
        ])
    raise ValueError("TIFF must contain a 3-D volume or a 4-D time series")


def open_volume_series(
    source: Union[str, Path],
    input_format: str = "auto",
    axis_order: str = "auto",
    frames_per_volume: int = 20,
    tiff_layout: str = "auto",
    z_range: Optional[Tuple[int, int]] = None,
    max_volumes: Optional[int] = None,
) -> VolumeSeries:
    """Open NPY/TIFF images or an HDF5 /volumes dataset without eager loading.

    Every VolumeFrame is one complete (Z,Y,X) volume. File time-series indices
    are local, zero-based IDs; they do not recover original acquisition IDs.
    """
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError("image source does not exist: %s" % path)
    selected_format = input_format.lower()
    if selected_format == "auto":
        if path.is_file():
            suffix = path.suffix.lower()
            selected_format = (
                "npy" if suffix == ".npy"
                else "h5" if suffix in (".h5", ".hdf5")
                else "tiff" if suffix in (".tif", ".tiff")
                else ""
            )
        else:
            has_npy = any(path.glob("*.npy"))
            has_tiff = any(path.glob("*.tif")) or any(path.glob("*.tiff"))
            if has_npy and has_tiff:
                raise ValueError(
                    "directory contains both NPY and TIFF; set input_format explicitly"
                )
            selected_format = "npy" if has_npy else "tiff" if has_tiff else ""
    if selected_format == "npy":
        return _npy_series(path, axis_order, z_range, max_volumes)
    if selected_format in ("h5", "hdf5"):
        return _h5_series(path, axis_order, z_range, max_volumes)
    if selected_format == "tiff":
        return _tiff_series(
            path, axis_order, frames_per_volume, tiff_layout, z_range, max_volumes
        )
    raise ValueError("could not detect NPY/HDF5/TIFF image format for %s" % path)

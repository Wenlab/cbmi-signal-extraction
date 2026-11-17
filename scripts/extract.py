"""修改下方分组参数后，运行 ``pixi run signal-extraction``。

每个时间点 T 对应一个完整的三维 volume，读取后统一为 (Z,Y,X)。
图像按 volume 读取，结果分块保存、支持续跑，最后合并为 intensity.h5。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from signal_extraction import (  # noqa: E402
    RegistrationCalibrationError,
    estimate_sampled_registration_matrix,
    extract_signals,
    load_coordinate_frames,
)
from signal_extraction.config import DEFAULT_CALIBRATION  # noqa: E402
from signal_extraction.images import open_volume_series  # noqa: E402


# ---------------------------------------------------------------------------
# 1. 输入路径
# ROI 的 volume ID 须与参考图像对应，坐标须匹配参考图像的 XY 几何及裁剪后的 Z 基准。
# ---------------------------------------------------------------------------
DATA_DIR = Path("data/dataset")  # 下方默认输入、输出路径的基目录。

# 神经元 ROI 框：NPY、dynamics/realtime H5，或实时结果 volume_*.h5 目录。
BOUNDING_BOX_SOURCE = DATA_DIR / "neuron_point_tuple.npy"
REFERENCE_IMAGES = DATA_DIR / "red_volumes.h5"  # 红通道；支持 NPY、H5、TIFF。
CALCIUM_IMAGES = DATA_DIR / "green_volumes.h5"  # 绿通道；设为 None 仅提取红通道。


# ---------------------------------------------------------------------------
# 2. 图像格式、轴顺序与 Z 层范围（红绿通道共用）
# T=完整 volume 的时间点，Z=深度层，Y=图像高，X=图像宽。
# ---------------------------------------------------------------------------
# "auto" 按文件扩展名或目录内容识别；也可指定 "npy"、"h5"、"tiff"。
INPUT_FORMAT = "auto"

# "tzyx"/"tyxz" 用于四维时间序列；"zyx"/"yxz" 用于单个三维 volume。
# "auto"：TIFF 优先读取轴元数据，否则按形状推断 Z 轴；H5 固定为 /volumes (T,Z,Y,X)。
AXIS_ORDER = "auto"

# 以下两个参数仅用于 TIFF 目录：
# "slices"：每个文件是一张二维 Z 切片；"volumes"：每个文件是一个三维 volume。
# "auto" 检查目录中的第一个 TIFF，判断是切片还是 volume。
TIFF_LAYOUT = "auto"

# 仅切片目录生效：原始采集每个 volume 的切片总数，用于先划分时间点。
FRAMES_PER_VOLUME = 20

# 所有格式均生效：None 保留全部 Z 层；(start, stop) 保留 [start, stop)，从 0 计数。
# 例：raw TIFF 每 volume 20 层、只取前 18 层：上面仍填 20，此处设 (0, 18)。
# 已裁好 Z 的导出数据通常保持 None；ROI 的 Z 坐标须对应裁剪后的 volume。
Z_RANGE = None


# ---------------------------------------------------------------------------
# 3. 时间范围与缺失数据（以下单位均为完整 volume）
# ---------------------------------------------------------------------------
# None 使用全部 ROI 时间点；整数 N 为 volume ID 的结束边界，仅处理 ID < N。
TOTAL_VOLUMES = None

# 主动跳过的 volume ID，例如 (12, 35)；ROI 时间轴中的这些位置保留为 NaN。
SKIP_VOLUMES = ()


# ---------------------------------------------------------------------------
# 4. 红绿通道配准（仅 CALCIUM_IMAGES 非 None 时生效）
# 有 NPY 路径：直接使用指定矩阵，跳过自动估计和失败回退。
# 无 NPY 路径：AUTO_REGISTRATION=False 直接用默认矩阵；
#              AUTO_REGISTRATION=True 先做 ECC，失败后由回退开关决定是否用默认矩阵。
# 默认矩阵与标定来源保存在 src/signal_extraction/config.py。
# ---------------------------------------------------------------------------
REGISTRATION_MATRIX_PATH = None  # None 或 Path(r"H:/.../matrix.npy")，矩阵形状为 (2,3)。
AUTO_REGISTRATION = True  # False 直接使用 config.py 默认矩阵；True 自动估计矩阵。
ALLOW_DEFAULT_CALIBRATION = True  # 仅自动模式失败时生效：True 回退默认矩阵，False 报错。

# 仅自动模式生效：从前 N 个未跳过的 volume 中抽样估计 ECC，结果供全部分块共用。
REGISTRATION_VOLUMES = 200

# True：矩阵目标坐标为水平翻转后的绿通道，提取时再映回原始绿通道。
# 当前默认矩阵要求 True，且原始 XY 为 1024×1024，未做 XY 裁剪、缩放或预对齐。
FLIP_CALCIUM_HORIZONTAL = True


# ---------------------------------------------------------------------------
# 5. 强度提取（红绿通道共用）
# ---------------------------------------------------------------------------
# 每个 Z 层在椭圆 ROI 内保留的最亮采样点比例，范围 (0,1]；0.5 表示最亮约 50%。
PERCENTILE = 0.5

# ROI 内椭圆采样范围的线性缩放系数，范围 (0,1]；1.0 使用 ROI 内接椭圆。
AREA_RATIO = 1.0

# 从提取强度中减去的固定背景值，单位与图像灰度一致；设为 0 则不扣除。
BACKGROUND_THRESHOLD = 102.0


# ---------------------------------------------------------------------------
# 6. 计算设备、输出与续跑
# ---------------------------------------------------------------------------
DEVICE = "auto"  # "auto" 优先使用可用 CUDA，否则 CPU；可指定 "cpu" 或 "cuda"。
OUTPUT = DATA_DIR / "intensity.h5"  # 最终合并结果。

# 每个输出分块最多包含的连续 volume 数；图像仍逐 volume 读取。
CHUNK_SIZE = 2000

# 保存分块、配准矩阵及配置记录；相同配置可续跑，配置不匹配会拒绝复用。
# 不同配置使用新的 OUTPUT/PARTS_DIR，避免混用结果。
PARTS_DIR = OUTPUT.parent / (OUTPUT.stem + "_parts")


def _image_series(source):
    return open_volume_series(
        source,
        input_format=INPUT_FORMAT,
        axis_order=AXIS_ORDER,
        frames_per_volume=FRAMES_PER_VOLUME,
        tiff_layout=TIFF_LAYOUT,
        z_range=Z_RANGE,
    )


def _extract_kwargs() -> dict:
    return dict(
        inference_result=BOUNDING_BOX_SOURCE,
        reference_images=REFERENCE_IMAGES,
        calcium_images=CALCIUM_IMAGES,
        device=DEVICE,
        percentile_reference=PERCENTILE,
        percentile_calcium=PERCENTILE,
        area_ratio=AREA_RATIO,
        background_threshold=BACKGROUND_THRESHOLD,
        input_format=INPUT_FORMAT,
        axis_order=AXIS_ORDER,
        tiff_layout=TIFF_LAYOUT,
        frames_per_volume=FRAMES_PER_VOLUME,
        z_range=Z_RANGE,
        frame_alignment="id",
        flip_calcium_horizontal=FLIP_CALCIUM_HORIZONTAL,
    )


def _chunk_ranges(frame_ids: np.ndarray) -> list:
    """Split at skipped/missing IDs so a part never bridges a temporal gap."""
    ranges = []
    skipped = set(SKIP_VOLUMES)
    for frame_id in frame_ids:
        value = int(frame_id)
        if value in skipped:
            continue
        if ranges and value == ranges[-1][1] and value - ranges[-1][0] < CHUNK_SIZE:
            ranges[-1] = (ranges[-1][0], value + 1)
        else:
            ranges.append((value, value + 1))
    return ranges


def _prepare_parts_dir(config_json: str) -> None:
    manifest = PARTS_DIR / "config.json"
    if manifest.exists():
        if manifest.read_text(encoding="utf-8") != config_json:
            raise ValueError(
                "Extraction settings changed; choose a new OUTPUT/PARTS_DIR "
                "to avoid reusing incompatible parts"
            )
    elif PARTS_DIR.exists() and any(PARTS_DIR.iterdir()):
        raise ValueError(
            "Existing parts have no configuration record; choose a new PARTS_DIR"
        )
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest.write_text(config_json, encoding="utf-8")


def _validate_registration_matrix(matrix) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise ValueError("Registration matrix must be finite shape (2,3)")
    return matrix


def _default_registration_matrix(reference_frame, calcium_frame) -> np.ndarray:
    if FLIP_CALCIUM_HORIZONTAL != DEFAULT_CALIBRATION["flip_calcium_horizontal"]:
        raise ValueError(
            "Default calibration does not match FLIP_CALCIUM_HORIZONTAL; "
            "provide a matching REGISTRATION_MATRIX_PATH or update config.py"
        )
    matrix = _validate_registration_matrix(DEFAULT_CALIBRATION["matrix"])
    expected = tuple(DEFAULT_CALIBRATION["image_shape_yx"])
    reference_shape = reference_frame.load().shape[1:]
    calcium_shape = calcium_frame.load().shape[1:]
    if reference_shape != expected or calcium_shape != expected:
        raise ValueError(
            "Default calibration requires image shape (Y,X) %r; got reference "
            "%r and calcium %r. Provide a matching REGISTRATION_MATRIX_PATH "
            "or update config.py" % (expected, reference_shape, calcium_shape)
        )
    return matrix


def _registration_matrix(reference, frame_ids: np.ndarray):
    if CALCIUM_IMAGES is None:
        return None
    if REGISTRATION_MATRIX_PATH is not None:
        return _validate_registration_matrix(
            np.load(REGISTRATION_MATRIX_PATH, allow_pickle=False)
        )
    matrix_path = PARTS_DIR / "registration_matrix.npy"
    diagnostics_path = PARTS_DIR / "registration_diagnostics.json"
    if matrix_path.exists():
        matrix = _validate_registration_matrix(
            np.load(matrix_path, allow_pickle=False)
        )
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        if diagnostics.get("strategy") == "fallback":
            warnings.warn(
                "Reusing cached default calibration after automatic channel "
                "registration failed: %s" % diagnostics["reason"],
                RuntimeWarning,
                stacklevel=2,
            )
        return matrix

    calcium = _image_series(CALCIUM_IMAGES).by_id
    reference_by_id = reference.by_id
    skipped = set(SKIP_VOLUMES)
    candidates = [
        int(value) for value in frame_ids if value not in skipped
    ][:REGISTRATION_VOLUMES if AUTO_REGISTRATION else 1]
    for frame_id in candidates:
        if frame_id not in reference_by_id or frame_id not in calcium:
            raise ValueError("Missing paired camera volume %d" % frame_id)
    if not AUTO_REGISTRATION:
        matrix = _default_registration_matrix(
            reference_by_id[candidates[0]], calcium[candidates[0]]
        )
        diagnostics = {
            "source": "default_calibration",
            "strategy": "default",
            "reason": "automatic registration disabled; configured default selected",
            "matrix": matrix.tolist(),
            "default_calibration": DEFAULT_CALIBRATION,
        }
        print(
            "Using configured default calibration; automatic registration is disabled."
        )
    else:
        try:
            matrix, diagnostics = estimate_sampled_registration_matrix(
                [reference_by_id[value].load for value in candidates],
                [calcium[value].load for value in candidates],
                frame_ids=candidates,
                flip_horizontal=FLIP_CALCIUM_HORIZONTAL,
            )
            matrix = _validate_registration_matrix(matrix)
            diagnostics = {
                **diagnostics,
                "source": "sampled_ecc_median",
                "reason": None,
            }
        except RegistrationCalibrationError as error:
            if not ALLOW_DEFAULT_CALIBRATION:
                raise
            matrix = _default_registration_matrix(
                reference_by_id[candidates[0]], calcium[candidates[0]]
            )
            diagnostics = {
                "source": "default_calibration",
                "strategy": "fallback",
                "reason": str(error),
                "matrix": matrix.tolist(),
                "default_calibration": DEFAULT_CALIBRATION,
                "automatic_registration_diagnostics": error.diagnostics,
            }
            warnings.warn(
                "Automatic channel registration failed (%s); using default "
                "calibration from %s."
                % (error, DEFAULT_CALIBRATION["source_file"]),
                RuntimeWarning,
                stacklevel=2,
            )
    temporary = matrix_path.with_suffix(".npy.partial")
    with temporary.open("wb") as handle:
        np.save(handle, matrix, allow_pickle=False)
    diagnostics_temporary = diagnostics_path.with_suffix(".json.partial")
    diagnostics_temporary.write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    diagnostics_temporary.replace(diagnostics_path)
    temporary.replace(matrix_path)
    return matrix


def _merge_parts(
    part_paths: list,
    frame_ids: np.ndarray,
    neuron_ids: np.ndarray,
    config_json: str,
    empty_frame_ids: np.ndarray,
) -> np.ndarray:
    channel_count = 1 if CALCIUM_IMAGES is None else 2
    activity = np.full(
        (channel_count, neuron_ids.size, frame_ids.size), np.nan, dtype=np.float32
    )
    z_start = np.full((neuron_ids.size, frame_ids.size), -1, dtype=np.int32)
    z_count = np.zeros((neuron_ids.size, frame_ids.size), dtype=np.int32)
    covered = np.zeros(frame_ids.size, dtype=bool)
    attributes = {}
    matrix = None
    for path in part_paths:
        with h5py.File(path, "r") as handle:
            if handle.attrs.get("script_config_json") != config_json:
                raise ValueError("Part settings do not match: %s" % path)
            part_frames = handle["frame_ids"][:]
            part_neurons = handle["neuron_ids"][:]
            if not np.isin(part_frames, frame_ids).all():
                raise ValueError("Part contains unexpected volume IDs: %s" % path)
            if not np.isin(part_neurons, neuron_ids).all():
                raise ValueError("Part contains unexpected neuron IDs: %s" % path)
            columns = np.searchsorted(frame_ids, part_frames)
            rows = np.searchsorted(neuron_ids, part_neurons)
            if covered[columns].any() or np.unique(columns).size != columns.size:
                raise ValueError("Parts contain duplicate volume IDs: %s" % path)
            activity[:, rows[:, None], columns[None, :]] = handle["activity"][:]
            z_start[rows[:, None], columns[None, :]] = handle["reference_z_start"][:]
            z_count[rows[:, None], columns[None, :]] = handle["reference_z_count"][:]
            covered[columns] = True
            if not attributes:
                attributes = dict(handle.attrs)
                if "registration_matrix" in handle:
                    matrix = handle["registration_matrix"][:]

    missing = frame_ids[
        ~covered
        & ~np.isin(frame_ids, SKIP_VOLUMES)
        & ~np.isin(frame_ids, empty_frame_ids)
    ]
    if missing.size:
        raise ValueError("Missing undeclared volumes after merge: %s" % missing[:10])
    parameters = json.loads(attributes["parameters_json"])
    parameters.update(
        start_volume=int(frame_ids[0]),
        max_volumes=int(frame_ids[-1] - frame_ids[0] + 1),
    )
    attributes["parameters_json"] = json.dumps(parameters)
    attributes["merged_from_parts"] = len(part_paths)
    attributes["skipped_volumes"] = np.asarray(SKIP_VOLUMES, dtype=np.int64)
    attributes["empty_roi_volumes"] = empty_frame_ids
    diagnostics_path = PARTS_DIR / "registration_diagnostics.json"
    if diagnostics_path.exists():
        attributes["registration_diagnostics_json"] = diagnostics_path.read_text(
            encoding="utf-8"
        )
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".partial")
    with h5py.File(temporary, "w") as handle:
        for name, values in (
            ("activity", activity),
            ("neuron_ids", neuron_ids),
            ("frame_ids", frame_ids),
            ("reference_z_start", z_start),
            ("reference_z_count", z_count),
        ):
            handle.create_dataset(
                name, data=values, compression="gzip" if values.ndim > 1 else None
            )
        if matrix is not None:
            handle.create_dataset("registration_matrix", data=matrix)
        for name, value in attributes.items():
            handle.attrs[name] = value
    temporary.replace(OUTPUT)
    return activity


def main() -> int:
    if CHUNK_SIZE <= 0:
        raise ValueError("CHUNK_SIZE must be positive")
    if (
        CALCIUM_IMAGES is not None
        and REGISTRATION_MATRIX_PATH is None
        and AUTO_REGISTRATION
        and REGISTRATION_VOLUMES <= 0
    ):
        raise ValueError("REGISTRATION_VOLUMES must be positive for automatic ECC")
    reference = _image_series(REFERENCE_IMAGES)
    frames = load_coordinate_frames(
        BOUNDING_BOX_SOURCE,
        expected_frame_ids=(
            [frame.frame_id for frame in reference.frames]
            if Path(BOUNDING_BOX_SOURCE).is_dir() else None
        ),
    )
    if TOTAL_VOLUMES is not None:
        frames = [frame for frame in frames if frame.frame_id < TOTAL_VOLUMES]
    frame_ids = np.asarray([frame.frame_id for frame in frames], dtype=np.int64)
    if frame_ids.size == 0 or (np.diff(frame_ids) <= 0).any():
        raise ValueError("ROI volume IDs must be non-empty and strictly increasing")
    neuron_ids = np.asarray(sorted({
        int(value) for frame in frames for value in frame.neuron_ids
    }), dtype=np.int64)
    has_neurons = np.asarray([frame.neuron_ids.size > 0 for frame in frames])
    # Empty ROI times remain in the final time axis without extracting images.
    empty_frame_ids = frame_ids[~has_neurons]
    ranges = _chunk_ranges(frame_ids[has_neurons])
    if not ranges or neuron_ids.size == 0:
        raise ValueError("No non-skipped volumes or valid neuron IDs to extract")

    config = {
        **_extract_kwargs(),
        "total_volumes": TOTAL_VOLUMES,
        "skip_volumes": SKIP_VOLUMES,
        "registration_matrix_path": REGISTRATION_MATRIX_PATH,
        "auto_registration": AUTO_REGISTRATION,
        "registration_volumes": REGISTRATION_VOLUMES,
        "allow_default_calibration": ALLOW_DEFAULT_CALIBRATION,
        "default_calibration": (
            DEFAULT_CALIBRATION
            if not AUTO_REGISTRATION or ALLOW_DEFAULT_CALIBRATION else None
        ),
    }
    config_json = json.dumps(config, default=str, sort_keys=True, indent=2)
    _prepare_parts_dir(config_json)
    matrix = _registration_matrix(reference, frame_ids)
    part_paths = []
    for start, stop in ranges:
        part_path = PARTS_DIR / ("part_%07d_%07d.h5" % (start, stop))
        part_paths.append(part_path)
        if part_path.exists():
            print("Skipping completed part: %s" % part_path.name)
            continue
        print("Extracting volumes [%d, %d) ..." % (start, stop))
        temporary = part_path.with_suffix(".h5.partial")
        extract_signals(
            output_path=temporary,
            start_volume=start,
            max_volumes=stop - start,
            registration_matrix=matrix,
            **_extract_kwargs(),
        )
        with h5py.File(temporary, "a") as handle:
            handle.attrs["script_config_json"] = config_json
        temporary.replace(part_path)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    activity = _merge_parts(
        part_paths, frame_ids, neuron_ids, config_json, empty_frame_ids
    )
    print("Saved %s: activity=%s" % (OUTPUT.resolve(), activity.shape))
    print("All-NaN volumes: %d / %d" % (
        np.isnan(activity).all(axis=(0, 1)).sum(), frame_ids.size
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

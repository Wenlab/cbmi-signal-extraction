# Signal extraction

这是一个不依赖 Treeformer 检测/配准模型的提信号仓库。它读取已经完成的
neuron bounding box 和原始 reference/calcium 图像，输出
`intensity.h5`。

CPU 和 GPU 使用同一个 Torch/TorchScript 算法：

1. 按 `[cx, cy, z*5, width, height, depth*5]` 批量建立 ROI；
2. `grid_sample` 采样椭圆 ROI；
3. 每个 Z 层保留最亮的指定百分比像素；
4. 搜索最亮的连续 3 层 Z 窗口；
5. calcium 使用 reference 找到的 Z 窗口；
6. 最终减去背景值。

CPU/GPU 仍可能存在很小的浮点差异。

## Pixi 环境

Linux 服务器默认安装 CUDA 11.8 版 PyTorch，Windows 默认安装 CPU 版 PyTorch：

```bash
cd /home/wangjinghao/JinghaoWang/signal_extraction
pixi install
pixi run signal-extraction --help
```

Windows PowerShell：

```powershell
cd path\to\signal_extraction
pixi install
pixi run signal-extraction --help
```

核心依赖是 `torch`、`numpy`、`h5py`、`tifffile`、OpenCV 和
`tqdm`。如果只处理单通道或直接提供配准矩阵，ECC 不会参与主计算。

## 双通道配准

默认 `registration_strategy="sample_median"`，其方法移植自
`CBMI_python_inference_slim_20260702/scripts/ecc_alignment_only.py`：

1. 从已与 inference 结果匹配的 reference/calcium volume 中，用固定随机种子
   `72` 无放回抽取最多 10 对（不足 10 对时全部使用）；
2. 每个 volume 使用当前读取范围内的全部 Z 层做 MIP，并先水平翻转 calcium
   MIP；
3. 每一对都从单位矩阵开始运行 OpenCV Euclidean ECC；
4. 丢弃 ECC correlation 小于 `0.1` 或求解失败的结果；
5. 对候选矩阵的旋转角、X 平移和 Y 平移分别取中位数；
6. 丢弃相对初步中位矩阵平移偏差超过 `3` 像素的离群结果；
7. 至少保留 3 个一致结果后再次取中位数，作为整个序列共用的配准矩阵。

若无法获得至少 3 个一致矩阵，默认会明确报错，而不会静默退回历史硬件矩阵。
这样可以避免把不可靠的标定用于整段信号。随机数种子、抽样数、correlation
阈值和一致结果数均可通过 CLI 调整。配准由 OpenCV 在 CPU 上完成；`--device`
只控制随后 Torch 提信号使用 CPU 还是 GPU。

兼容模式 `--registration-strategy single` 只用 `--registration-frame` 指定的
一个 volume 做全 Z-MIP ECC；标定失败时会明确报错，不使用默认矩阵回退。

## Neuron bounding box 输入

`--bounding-box-source`（兼容别名：`--inference-result`）支持三类输入：

1. slim 仓库产生的 `dynamics.h5`：读取根目录下自然排序的逐-volume group，
   优先读取 `d_neuron_pt_tuple_matched_raw_vol`，也兼容 `neuron_pt_tuple`；行号
   作为稳定 neuron ID。
2. `neuron_pt_tuple.npy`：支持 `(T,N,F)` 或单帧 `(N,F)`，行号作为稳定
   neuron ID。
3. realtime bounding-box result：既支持一个聚合 HDF5，也支持
   `roi_source.py` 使用的目录形式。目录中每个文件必须命名为
   `volume_<id>.h5`，从
   `/real-time_analysis/volume_<8位id>/neuron_pt_tuple` 读取 box，并从
   `neuron_pred_ids` 读取稳定 ID；负 ID 被过滤。提信号 pipeline 会以 reference
   图像的 volume ID 为期望时间轴，因此缺失或不含有效结果的 realtime volume
   会保留为空时间点，输出相应位置为 `NaN`，不会将后续帧前移。

默认 `--bounding-box-format auto` 会根据目录、后缀和 HDF5 内容自动判断；也可
显式指定 `dynamics`、`npy` 或 `realtime-results`。

坐标约定保持现有 CBMI 代码不变：

```text
[cx, cy, z*5, width, height, depth*5, ...]
```

## 图像输入

reference 和 calcium 可独立指定为 NPY 或 TIFF：

- 一个目录，每个 `.npy` 是一个 `(Z,Y,X)` volume；
- 一个 `.npy` 文件，形状为 `(T,Z,Y,X)`；
- 一个目录，每个多页 TIFF 是一个 volume：`--tiff-layout volumes`；
- 一个目录，每个 TIFF 是一张 Z slice：`--tiff-layout slices`，用
  `--frames-per-volume` 分组；
- 一个 3-D/4-D TIFF 文件。

默认裁取 `Z=[0,18)`，与当前 slim 配置一致。若图像已经裁好或需要全部 Z 层，
CLI 使用 `--z-stop -1`，Python API 使用 `z_range=None`。

自动轴顺序适合常见 `(Z,Y,X)`。不确定时应显式传：

```text
--axis-order zyx    # 每个文件一个 volume
--axis-order tzyx   # 单文件时间序列
--axis-order yxz / tyxz
```

## CLI 示例

### CPU、双通道、每个 volume 一个 NPY

```bash
pixi run signal-extraction \
  --bounding-box-source results/dynamics.h5 \
  --reference-images data/reference_npy \
  --calcium-images data/calcium_npy \
  --output results/intensity.h5 \
  --device cpu \
  --input-format npy \
  --axis-order zyx
```

也可以不安装命令入口而直接运行：

```bash
pixi run signal-extraction --help
```

### GPU、realtime HDF5、连续单帧 TIFF

```bash
pixi run signal-extraction \
  --bounding-box-source realtime_results_directory \
  --bounding-box-format realtime-results \
  --reference-images data/reference_tiff \
  --calcium-images data/calcium_tiff \
  --output intensity.h5 \
  --device cuda \
  --input-format tiff \
  --tiff-layout slices \
  --frames-per-volume 20
```

### 只提 reference

```bash
pixi run signal-extraction \
  --bounding-box-source neuron_pt_tuple.npy \
  --reference-images reference.npy \
  --output reference_intensity.h5 \
  --device auto \
  --z-stop -1
```

### 已经有标定矩阵

```bash
pixi run signal-extraction ... --registration-matrix registration_matrix.npy
```

传入矩阵时不会运行 ECC。`--no-ecc` 只让仿射矩阵变为单位矩阵；默认的 calcium
水平镜像仍保留。只有确认两个通道图像方向相同，才同时添加
`--no-calcium-horizontal-flip`。

### 调整随机多 volume 配准

```bash
pixi run signal-extraction ... \
  --registration-strategy sample_median \
  --registration-num-samples 10 \
  --registration-seed 72 \
  --registration-min-correlation 0.1 \
  --registration-min-successful-samples 3
```

`--max-translation-deviation-px` 控制相对初步中位矩阵的平移离群筛选阈值。

## Python API

```python
from signal_extraction import extract_signals

result = extract_signals(
    inference_result="dynamics.h5",
    reference_images="reference_npy",
    calcium_images="calcium_npy",
    output_path="intensity.h5",
    device="cpu",             # "auto", "cpu", "cuda", "cuda:1"
    percentile_reference=0.01,
    percentile_calcium=0.01,
    background_threshold=102,
    axis_order="zyx",
    registration_strategy="sample_median",  # 默认：随机最多 10 个 volume
)

print(result.activity.shape)  # (2, neurons, time)
```

单个 volume 的低层接口：

```python
from signal_extraction import extract_volume_signals

one_frame = extract_volume_signals(
    volume_zyx,
    coords_n_by_f,
    device="cpu",
    percentile=0.01,
    background_threshold=102,
)
```

## 输出

HDF5 包含：

- `activity`：`(C,N,T)`；单通道 `C=1`，双通道 `C=2`；
- `neuron_ids`：activity 的 N 轴对应的 inference ID；
- `frame_ids`：activity 的 T 轴对应的 volume ID；
- `reference_z_start`、`reference_z_count`：reference 最佳窗口；
- `registration_matrix`：双通道坐标变换矩阵。

输出属性还记录设备、算法名、提取参数和配准诊断，便于复现实验。

缺失 ID、负 ID 或无效坐标写为 `NaN`，不会默认为零。

## 帧匹配

默认 `frame_alignment="auto"`：优先用 volume ID 匹配；如果 ID 无法全部匹配但
两边数量相同，报警后按自然排序位置匹配。正式数据推荐使用
`--frame-alignment id`，这样缺帧会立即报错。

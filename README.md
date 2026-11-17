# CBMI Signal Extraction

Extract per-neuron fluorescence from existing bounding-box tracks and volumetric
reference (red) / calcium (green) images on CPU or CUDA. No detection model is
required. Output values are fluorescence after constant background subtraction.

## Setup and run

Use Python 3.11. For a CPU environment:

```bash
python -m pip install "numpy>=1.26,<3" "h5py>=3.10,<4" "hdf5plugin>=6,<8" "tifffile>=2024.8,<2027" "opencv-python>=4.10,<5" "tqdm>=4.66,<5"
python -m pip install "torch==2.6.*" --index-url https://download.pytorch.org/whl/cpu
```

For GPU extraction, use a CUDA-enabled PyTorch build instead of the CPU build.

Edit the settings at the top of `scripts/extract.py`:

```python
DATA_DIR = Path("data/dataset")
BOUNDING_BOX_SOURCE = DATA_DIR / "neuron_point_tuple.npy"
REFERENCE_IMAGES = DATA_DIR / "red_volumes.h5"
CALCIUM_IMAGES = DATA_DIR / "green_volumes.h5"  # None for reference only.
OUTPUT = DATA_DIR / "intensity.h5"
```

Run from the repository root:

```bash
python scripts/extract.py
```

This is the only extraction entry point; settings are edited in the script.
If the local Pixi manifest is available, `pixi run signal-extraction` runs the
same script.

## Image and ROI inputs

Images use `(T, Z, Y, X)`. Each time point is one complete volume, not one
Z slice. Each volume is loaded as `(Z, Y, X)`.

| Input | Layout and reading |
| --- | --- |
| NPY file | `(T,Z,Y,X)` or one `(Z,Y,X)` volume; read-only memory mapping. |
| NPY directory | One 3-D volume per file; volume IDs come from filenames. |
| H5 / HDF5 file | Fixed `/volumes` dataset with shape `(T,Z,Y,X)`; read one volume at a time. Zstd compression requires `hdf5plugin`. |
| TIFF file | One 3-D volume or a 4-D series; memory mapping when possible, otherwise decoding by volume. |
| TIFF directory | One 3-D volume per file, or 2-D slice files grouped into volumes. |

Use an explicit image file path when the directory also contains ROI or mask
files. The two channels may use different formats with `INPUT_FORMAT="auto"`.

| Setting | Meaning |
| --- | --- |
| `INPUT_FORMAT="auto"` | Detect `npy`, `h5`, or `tiff` from each source. |
| `AXIS_ORDER="auto"` | Use supported TIFF axis metadata or an NPY shape heuristic. Explicit orders: `zyx`, `yxz`, `tzyx`, `tyxz`. H5 always uses TZYX. |
| `TIFF_LAYOUT="auto"` | Detect `slices` or `volumes` for a TIFF directory. |
| `FRAMES_PER_VOLUME=20` | Number of 2-D TIFF slice files per complete volume; only used for slice directories. |
| `Z_RANGE=None` | Read all Z slices. `(start, stop)` selects a range with an exclusive stop. ROI Z coordinates must match the selected image coordinates. |

ROI inputs can be coordinate NPY arrays, dynamics HDF5, aggregate realtime
HDF5, or a directory of realtime `volume_<id>.h5` results. Coordinate NPY arrays
use `(T,N,F)` or `(N,F)` for one volume. The standard bounding-box fields are:

```text
[cx, cy, z * 5, width, height, depth * 5, ...]
```

XY coordinates are zero-based pixel centers in the reference image. Z and depth
are divided by 5 to obtain slice units. Use ROI coordinates exported with the
same cropping, flips, and scaling as the images.

The script matches ROI and image volume IDs. Single-file image series use local
IDs `0..T-1`; subset exports need ROI tracks in the same order. Voxel size, Z step,
volume sampling interval, and the original acquisition-ID mapping must be
recorded separately.

## Channel registration

Registration is only used for dual-channel extraction.

| `REGISTRATION_MATRIX_PATH` | `AUTO_REGISTRATION` | Behavior |
| --- | --- | --- |
| Path to an existing NPY | Either | Use the supplied matrix without automatic calibration. |
| `None` | `False` | Use `DEFAULT_CALIBRATION` from `src/signal_extraction/config.py`. |
| `None` | `True` | Estimate a matrix with ECC; use the default calibration if ECC calibration fails. |

To use the default calibration directly:

```python
REGISTRATION_MATRIX_PATH = None
AUTO_REGISTRATION = False
FLIP_CALCIUM_HORIZONTAL = True
```

With `FLIP_CALCIUM_HORIZONTAL=True`, the finite `(2,3)` matrix has this meaning:

> Maps reference (red) XY coordinates to horizontally flipped green XY coordinates.

```python
xf, yf = matrix @ [x_reference, y_reference, 1]
x_green = green_width - 1 - xf
y_green = yf
```

Extraction maps ROI centers into the original green image and samples that
image. The matrix acts on XY only and is shared across all volumes and Z slices.
The horizontal flip is applied separately from the stored matrix.

Automatic calibration samples up to 10 paired volumes from the initial
`REGISTRATION_VOLUMES=200` candidates, using seed 72. It estimates Euclidean ECC
transforms from Z projections and requires at least three consistent estimates.
Image-reading errors, invalid inputs, and missing dependencies still raise
errors; only calibration failures trigger the default fallback.

The default matrix comes from the November 4, 2026 camera calibration and requires
original `1024 x 1024` images with `FLIP_CALCIUM_HORIZONTAL=True`. It assumes no XY
cropping, resizing, or prior alignment. Update the matrix and its metadata together
when recalibrating the cameras. Supplied matrices must match the selected image
geometry and flip convention.

## Extraction settings

| Setting | Default | Meaning |
| --- | --- | --- |
| `PERCENTILE` | `0.5` | Brightest fraction of sampled ROI values retained per Z slice. |
| `AREA_RATIO` | `1.0` | Linear scale of the ellipse inscribed in each bounding box. |
| `BACKGROUND_THRESHOLD` | `102.0` | Constant subtracted from extracted intensity. |
| `DEVICE` | `"auto"` | CUDA when available, otherwise CPU; also accepts `"cpu"` or `"cuda"`. |
| `TOTAL_VOLUMES` | `None` | All ROI times; an integer keeps volume IDs below that value. |
| `SKIP_VOLUMES` | `()` | Explicitly skipped volume IDs; retain NaN on the ROI time axis. |
| `CHUNK_SIZE` | `2000` | Maximum number of contiguous volumes per output part. |

The reference channel selects the brightest contiguous three-slice window inside
each ROI, or the available slices for a shallower ROI. The calcium channel uses
the same Z window. CPU and CUDA use the same Torch algorithm, with possible small
floating-point differences.

## Output and resuming

The main output is `intensity.h5`:

| Dataset | Shape / meaning |
| --- | --- |
| `activity` | `(C,N,T)`: channel, neuron, complete volume. Channels are reference, then calcium; single-channel output has `C=1`. |
| `neuron_ids` | IDs corresponding to the N axis. |
| `frame_ids` | Volume IDs corresponding to the T axis. |
| `reference_z_start`, `reference_z_count` | `(N,T)` reference Z-window positions and slice counts. |
| `registration_matrix` | The applied `(2,3)` matrix for dual-channel extraction. |

Missing neurons, invalid ROIs, empty ROI times, and explicitly skipped times are
preserved as NaN. HDF5 attributes record extraction settings, channel names,
device, and registration diagnostics.

Completed parts and registration are cached in `PARTS_DIR`. Interrupted
`.partial` files are recomputed; completed parts are reused when the configuration
matches. Final merging aligns neuron IDs and volume IDs across parts.

Configuration changes require a new `OUTPUT` / `PARTS_DIR`. Use a new destination
also when replacing input file contents at the same path. Extraction reads source
images and ROI tracks without modifying them.

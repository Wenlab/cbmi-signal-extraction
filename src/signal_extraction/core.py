"""CPU/GPU implementation of the current improved signal extractor.

The numerical path intentionally mirrors
``src/online_inference/improved_signal_extract.py``.  CPU and CUDA therefore
use the same grid sampling, percentile thresholding and Z-window operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F


@torch.jit.script
def _extract_neuron_intensities_jit(
    volume_3d: torch.Tensor,
    neuron_pt_tuple: torch.Tensor,
    percentile: float,
    area_ratio: float,
    forced_z_start: Optional[torch.Tensor] = None,
    depth_correction: int = 0,
):
    """Vectorised grid_sample + percentile threshold + sliding Z-window."""
    device = volume_3d.device
    vol_z, vol_y, vol_x = volume_3d.shape
    vol_batch = volume_3d.unsqueeze(0).unsqueeze(0)

    n_neurons = neuron_pt_tuple.shape[0]
    cx = neuron_pt_tuple[:, 0]
    cy = neuron_pt_tuple[:, 1]
    z_raw = neuron_pt_tuple[:, 2]
    width = neuron_pt_tuple[:, 3]
    height = neuron_pt_tuple[:, 4]

    if neuron_pt_tuple.shape[1] > 5:
        depth_raw = neuron_pt_tuple[:, 5]
    else:
        depth_raw = torch.full((n_neurons,), 10.0, device=device)

    z_center = z_raw / 5.0
    depth = (depth_raw / 5.0).int()
    z_min = torch.ceil(z_center - depth / 2.0).int()
    z_max = torch.ceil(z_center + depth / 2.0).int()
    z_min = torch.clamp(z_min, min=0)
    z_max = torch.clamp(z_max, max=vol_z) + depth_correction
    depth_int = torch.clamp(z_max - z_min, min=1)
    max_depth = int(depth_int.max().item())

    max_width = int(width.max().item()) if width.numel() > 0 else 16
    max_height = int(height.max().item()) if height.numel() > 0 else 16
    sample_x = max(16, min(max_width, 128))
    sample_y = max(16, min(max_height, 128))

    range_z = torch.arange(max_depth, device=device, dtype=torch.float32)
    z_coords = z_min.unsqueeze(1) + range_z.unsqueeze(0)
    valid_z_mask = range_z.unsqueeze(0) < depth_int.unsqueeze(1)
    z_norm = (2.0 * z_coords / (vol_z - 1.0)) - 1.0

    range_y = torch.linspace(-0.5, 0.5, sample_y, device=device)
    range_x = torch.linspace(-0.5, 0.5, sample_x, device=device)
    y_grid = (
        cy.view(n_neurons, 1, 1)
        + range_y.view(1, sample_y, 1) * height.view(n_neurons, 1, 1)
    )
    x_grid = (
        cx.view(n_neurons, 1, 1)
        + range_x.view(1, 1, sample_x) * width.view(n_neurons, 1, 1)
    )
    y_norm = (2.0 * y_grid / (vol_y - 1.0)) - 1.0
    x_norm = (2.0 * x_grid / (vol_x - 1.0)) - 1.0

    z_expanded = z_norm.view(n_neurons, max_depth, 1, 1).expand(
        n_neurons, max_depth, sample_y, sample_x
    )
    y_expanded = y_norm.view(n_neurons, 1, sample_y, 1).expand(
        n_neurons, max_depth, sample_y, sample_x
    )
    x_expanded = x_norm.view(n_neurons, 1, 1, sample_x).expand(
        n_neurons, max_depth, sample_y, sample_x
    )
    grid = torch.stack((x_expanded, y_expanded, z_expanded), dim=-1)

    samples = F.grid_sample(
        vol_batch.expand(n_neurons, -1, -1, -1, -1),
        grid,
        align_corners=True,
        padding_mode="zeros",
    ).squeeze(1)

    y_rel = range_y.view(1, sample_y, 1)
    x_rel = range_x.view(1, 1, sample_x)
    mask_2d = (y_rel ** 2 + x_rel ** 2) <= (0.5 * area_ratio) ** 2
    samples_flat = samples.view(n_neurons, max_depth, -1)
    masked_samples = samples_flat[:, :, mask_2d.view(-1)]

    sums_all = masked_samples.sum(dim=-1)
    pixels_per_slice = masked_samples.shape[-1]
    sums_above = torch.zeros((n_neurons, max_depth), device=device)
    counts_above = torch.zeros((n_neurons, max_depth), device=device)

    if percentile < 1.0:
        thresholds = torch.quantile(
            masked_samples, 1.0 - percentile, dim=-1
        )
        mask_above = masked_samples >= thresholds.unsqueeze(-1)
        sums_above = (masked_samples * mask_above).sum(dim=-1)
        counts_above = mask_above.sum(dim=-1).float()
        sums_above = sums_above * valid_z_mask
        counts_above = counts_above * valid_z_mask

    sums_all = sums_all * valid_z_mask
    kernel = torch.ones((1, 1, 3), device=device)

    sums_all_win = F.conv1d(
        F.pad(sums_all.unsqueeze(1), (0, 2)), kernel
    ).squeeze(1)
    counts_all = (
        torch.full(
            (n_neurons, max_depth), float(pixels_per_slice), device=device
        )
        * valid_z_mask
    )
    counts_all_win = F.conv1d(
        F.pad(counts_all.unsqueeze(1), (0, 2)), kernel
    ).squeeze(1)
    avg_all = sums_all_win / torch.clamp_min(counts_all_win, 1e-6)

    if percentile < 1.0:
        sums_above_win = F.conv1d(
            F.pad(sums_above.unsqueeze(1), (0, 2)), kernel
        ).squeeze(1)
        counts_above_win = F.conv1d(
            F.pad(counts_above.unsqueeze(1), (0, 2)), kernel
        ).squeeze(1)
        avg_primary = sums_above_win / torch.clamp_min(
            counts_above_win, 1e-6
        )
        avg_window = torch.where(
            counts_above_win > 0, avg_primary, avg_all
        )
    else:
        avg_window = avg_all

    window_index = torch.arange(max_depth, device=device).unsqueeze(0)
    depth_tensor = depth_int.unsqueeze(1)
    complete_three_slice = (
        (depth_tensor >= 3) & (window_index <= depth_tensor - 3)
    )
    short_depth = (depth_tensor < 3) & (window_index == 0)
    valid_window_mask = complete_three_slice | short_depth
    avg_window[~valid_window_mask] = -float("inf")

    if forced_z_start is not None:
        forced_index = (forced_z_start - z_min).long()
        forced_index = torch.clamp(
            forced_index, min=0, max=int(max_depth) - 1
        )
        max_values = avg_window.gather(
            1, forced_index.unsqueeze(1)
        ).squeeze(1)
        valid_forced = (
            (forced_z_start >= 0) & (max_values > -float("inf"))
        )
        max_values = torch.where(
            valid_forced,
            max_values,
            torch.tensor(-float("inf"), device=device),
        )
        valid_neuron = max_values > -float("inf")
        return max_values, forced_index, z_min, depth_int, valid_neuron

    max_values, max_indices = torch.max(avg_window, dim=1)
    valid_neuron = max_values > -float("inf")
    return max_values, max_indices, z_min, depth_int, valid_neuron


@dataclass
class VolumeSignalResult:
    """Signal extraction result for one volume and one channel."""

    intensities: np.ndarray
    z_start: np.ndarray
    z_count: np.ndarray
    valid: np.ndarray


def resolve_device(device: Union[str, torch.device] = "auto") -> torch.device:
    """Resolve ``auto``, ``cpu`` or a CUDA device with clear validation."""
    if isinstance(device, torch.device):
        resolved = device
    elif str(device).lower() == "auto":
        resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved = torch.device(device)

    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False. "
            "Use device='cpu' or install a CUDA-enabled PyTorch build."
        )
    return resolved


def _as_volume_tensor(
    volume: Union[np.ndarray, torch.Tensor], device: torch.device
) -> torch.Tensor:
    if isinstance(volume, torch.Tensor):
        tensor = volume
    else:
        array = np.asarray(volume)
        if array.ndim != 3:
            raise ValueError(
                "volume must have shape (Z, Y, X), got %r" % (array.shape,)
            )
        tensor = torch.from_numpy(np.ascontiguousarray(array))
    if tensor.ndim == 4 and tensor.shape[1] == 1:
        tensor = tensor.squeeze(1)
    if tensor.ndim != 3:
        raise ValueError(
            "volume must have shape (Z, Y, X) or (Z, 1, Y, X), got %r"
            % (tuple(tensor.shape),)
        )
    if min(tensor.shape) < 2:
        raise ValueError("all volume dimensions must be at least 2")
    return tensor.to(device=device, dtype=torch.float32)


def extract_volume_signals(
    volume: Union[np.ndarray, torch.Tensor],
    neuron_pt_tuple: Union[np.ndarray, torch.Tensor],
    percentile: float = 0.01,
    area_ratio: float = 1.0,
    background_threshold: float = 102.0,
    device: Union[str, torch.device] = "auto",
    forced_z_start: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> VolumeSignalResult:
    """Extract all neuron signals from one ``(Z,Y,X)`` volume.

    Coordinates use ``[cx, cy, z*5, width, height, depth*5, ...]``. Invalid
    coordinate rows are retained in the output as ``NaN`` rather than being
    sent through ``grid_sample``.
    """
    if not 0.0 < float(percentile) <= 1.0:
        raise ValueError("percentile must be in (0, 1]")
    if not 0.0 < float(area_ratio) <= 1.0:
        raise ValueError("area_ratio must be in (0, 1]")

    resolved_device = resolve_device(device)
    volume_tensor = _as_volume_tensor(volume, resolved_device)
    coords_np = (
        neuron_pt_tuple.detach().cpu().numpy()
        if isinstance(neuron_pt_tuple, torch.Tensor)
        else np.asarray(neuron_pt_tuple)
    )
    if coords_np.ndim != 2 or coords_np.shape[1] < 5:
        raise ValueError(
            "neuron_pt_tuple must have shape (N,F), F>=5; got %r"
            % (coords_np.shape,)
        )

    n_neurons = coords_np.shape[0]
    intensities = np.full(n_neurons, np.nan, dtype=np.float32)
    z_start = np.full(n_neurons, -1, dtype=np.int32)
    z_count = np.zeros(n_neurons, dtype=np.int32)
    valid_output = np.zeros(n_neurons, dtype=bool)
    if n_neurons == 0:
        return VolumeSignalResult(intensities, z_start, z_count, valid_output)

    required_columns = min(coords_np.shape[1], 6)
    valid_input = np.isfinite(coords_np[:, :required_columns]).all(axis=1)
    valid_input &= coords_np[:, 3] > 0
    valid_input &= coords_np[:, 4] > 0
    valid_indices = np.flatnonzero(valid_input)
    if valid_indices.size == 0:
        return VolumeSignalResult(intensities, z_start, z_count, valid_output)

    coords_tensor = torch.as_tensor(
        np.ascontiguousarray(coords_np[valid_indices]),
        device=resolved_device,
        dtype=torch.float32,
    )

    forced_tensor = None
    if forced_z_start is not None:
        forced_np = (
            forced_z_start.detach().cpu().numpy()
            if isinstance(forced_z_start, torch.Tensor)
            else np.asarray(forced_z_start)
        )
        if forced_np.shape != (n_neurons,):
            raise ValueError(
                "forced_z_start must have shape (%d,), got %r"
                % (n_neurons, forced_np.shape)
            )
        forced_valid = np.asarray(forced_np[valid_indices], dtype=np.float32)
        forced_valid = np.where(np.isfinite(forced_valid), forced_valid, -1.0)
        forced_tensor = torch.as_tensor(
            forced_valid, device=resolved_device, dtype=torch.float32
        )

    with torch.inference_mode():
        values, indices, z_min, depth_int, valid_neuron = (
            _extract_neuron_intensities_jit(
                volume_tensor,
                coords_tensor,
                float(percentile),
                float(area_ratio),
                forced_tensor,
            )
        )

    values_np = (values - float(background_threshold)).cpu().numpy()
    valid_np = valid_neuron.cpu().numpy().astype(bool, copy=False)
    starts_np = (z_min + indices).cpu().numpy().astype(np.int32, copy=False)
    counts_np = torch.minimum(
        depth_int, torch.full_like(depth_int, 3)
    ).cpu().numpy().astype(np.int32, copy=False)

    output_indices = valid_indices[valid_np]
    intensities[output_indices] = values_np[valid_np]
    z_start[output_indices] = starts_np[valid_np]
    z_count[output_indices] = counts_np[valid_np]
    valid_output[output_indices] = True
    return VolumeSignalResult(intensities, z_start, z_count, valid_output)


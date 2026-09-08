"""Command-line interface for signal extraction."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from .pipeline import extract_signals


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract CBMI reference/calcium signals from neuron bounding boxes "
            "using the GPU-improved-compatible Torch path."
        )
    )
    parser.add_argument(
        "--bounding-box-source",
        "--inference-result",
        dest="inference_result",
        metavar="BOUNDING_BOX_SOURCE",
        required=True,
        help=(
            "dynamics.h5, neuron_pt_tuple.npy, aggregate realtime HDF5, or "
            "a realtime volume_*.h5 directory"
        ),
    )
    parser.add_argument(
        "--bounding-box-format",
        choices=("auto", "dynamics", "npy", "realtime-results"),
        default="auto",
    )
    parser.add_argument("--reference-images", required=True)
    parser.add_argument("--calcium-images")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:1, ...")
    parser.add_argument("--reference-percentile", type=float, default=0.01)
    parser.add_argument("--calcium-percentile", type=float, default=0.01)
    parser.add_argument("--area-ratio", type=float, default=1.0)
    parser.add_argument("--background-threshold", type=float, default=102.0)
    parser.add_argument("--input-format", choices=("auto", "npy", "tiff"), default="auto")
    parser.add_argument(
        "--axis-order",
        choices=("auto", "zyx", "yxz", "tzyx", "tyxz"),
        default="auto",
    )
    parser.add_argument("--frames-per-volume", type=int, default=20)
    parser.add_argument(
        "--tiff-layout",
        choices=("auto", "slices", "volumes"),
        default="auto",
    )
    parser.add_argument("--z-start", type=int, default=0)
    parser.add_argument(
        "--z-stop",
        type=int,
        default=18,
        help="Exclusive Z stop; set a negative value to disable Z slicing",
    )
    parser.add_argument(
        "--frame-alignment", choices=("auto", "id", "position"), default="auto"
    )
    parser.add_argument("--no-ecc", action="store_true")
    parser.add_argument(
        "--registration-strategy",
        choices=("sample_median", "single"),
        default="sample_median",
        help=(
            "sample_median uses random paired volumes (default); single uses "
            "--registration-frame"
        ),
    )
    parser.add_argument("--registration-num-samples", type=int, default=10)
    parser.add_argument("--registration-seed", type=int, default=72)
    parser.add_argument("--registration-min-correlation", type=float, default=0.1)
    parser.add_argument(
        "--registration-min-successful-samples", type=int, default=3
    )
    parser.add_argument("--registration-frame", type=int, default=0)
    parser.add_argument("--registration-matrix")
    parser.add_argument("--max-translation-deviation-px", type=float, default=3.0)
    parser.add_argument("--no-calcium-horizontal-flip", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    z_range = None if args.z_stop < 0 else (args.z_start, args.z_stop)
    result = extract_signals(
        inference_result=args.inference_result,
        reference_images=args.reference_images,
        calcium_images=args.calcium_images,
        output_path=args.output,
        device=args.device,
        percentile_reference=args.reference_percentile,
        percentile_calcium=args.calcium_percentile,
        area_ratio=args.area_ratio,
        background_threshold=args.background_threshold,
        bounding_box_format=args.bounding_box_format,
        input_format=args.input_format,
        axis_order=args.axis_order,
        frames_per_volume=args.frames_per_volume,
        tiff_layout=args.tiff_layout,
        z_range=z_range,
        frame_alignment=args.frame_alignment,
        register_calcium=not args.no_ecc,
        registration_strategy=args.registration_strategy,
        registration_num_samples=args.registration_num_samples,
        registration_seed=args.registration_seed,
        registration_min_correlation=args.registration_min_correlation,
        registration_min_successful_samples=(
            args.registration_min_successful_samples
        ),
        registration_frame=args.registration_frame,
        registration_matrix=args.registration_matrix,
        max_translation_deviation_px=args.max_translation_deviation_px,
        flip_calcium_horizontal=not args.no_calcium_horizontal_flip,
        show_progress=not args.no_progress,
    )
    print(
        "Saved %s: activity=%s, neurons=%d, frames=%d, device=%s"
        % (
            Path(args.output).resolve(),
            result.activity.shape,
            result.neuron_ids.size,
            result.frame_ids.size,
            result.device,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

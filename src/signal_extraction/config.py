"""Default two-camera calibration for automatic registration failures.

The matrix maps original red (x, y) pixel centers to horizontally flipped
green (x, y). Extraction then maps x back with ``x = width - 1 - x``.
It applies to the original 1024 x 1024 camera geometry, without XY cropping,
resizing or prior alignment. Update the matrix and metadata together when
recalibrating the cameras.
"""

DEFAULT_CALIBRATION = {
    "matrix": [
        [1.0009169398394704, 0.01525278227410993, -8.48801270091235],
        [-0.01525278227410993, 1.0009169398394704, 8.073503145016172],
    ],
    "image_shape_yx": [1024, 1024],
    "flip_calcium_horizontal": True,
    "direction": "reference_to_flipped_calcium",
    "coordinate_order": "xy",
    "coordinate_origin": "zero_based_pixel_centers",
    "source_file": r"H:\Process_temporary\cali_2026-11-04_15-10-16\calib_params.json",
    "source_sha256": "4bf7e7691e84d9921ab5756ee6b30e19d6c1870bfec8ba6fcf5c23988d6b470d",
    "source_direction": "flipped_reference_to_calcium",
    "conversion": "E = F_green @ M_source @ F_red",
}

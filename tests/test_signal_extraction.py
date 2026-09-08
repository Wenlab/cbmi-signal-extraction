import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import tifffile
import torch

from signal_extraction import extract_signals, extract_volume_signals
from signal_extraction.coordinates import load_coordinate_frames
from signal_extraction.images import open_volume_series
from signal_extraction.registration import (
    RegistrationCalibrationError,
    estimate_registration_matrix,
    estimate_sampled_registration_matrix,
)


def _euclidean_matrix(angle_deg, tx, ty):
    angle = np.deg2rad(angle_deg)
    return np.asarray(
        [
            [np.cos(angle), -np.sin(angle), tx],
            [np.sin(angle), np.cos(angle), ty],
        ],
        dtype=np.float32,
    )


class SignalExtractionTests(unittest.TestCase):
    def test_volume_cpu(self):
        rng = np.random.default_rng(72)
        volume = rng.integers(80, 1000, size=(18, 48, 52), dtype=np.uint16)
        coords = np.array(
            [[18.2, 20.7, 45, 10, 12, 20], [34, 30, 55, 8, 8, 15]],
            dtype=np.float32,
        )
        result = extract_volume_signals(volume, coords, device="cpu")
        self.assertEqual(result.intensities.shape, (2,))
        self.assertTrue(np.isfinite(result.intensities).all())
        self.assertTrue((result.z_count == 3).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cpu_cuda_close(self):
        rng = np.random.default_rng(7)
        volume = rng.integers(80, 1000, size=(18, 48, 52), dtype=np.uint16)
        coords = np.array(
            [[18.2, 20.7, 45, 10, 12, 20], [34, 30, 55, 8, 8, 15]],
            dtype=np.float32,
        )
        cpu = extract_volume_signals(volume, coords, device="cpu")
        cuda = extract_volume_signals(volume, coords, device="cuda")
        np.testing.assert_allclose(
            cpu.intensities, cuda.intensities, rtol=2e-5, atol=2e-3
        )
        np.testing.assert_array_equal(cpu.z_start, cuda.z_start)

    def test_realtime_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "realtime.h5"
            with h5py.File(path, "w") as handle:
                root = handle.create_group("real-time_analysis")
                for frame_id, neuron_ids in ((5, [10, 30]), (7, [30, 42])):
                    group = root.create_group("volume_%08d" % frame_id)
                    group["neuron_pt_tuple"] = np.ones((2, 8), dtype=np.float32)
                    group["neuron_pred_ids"] = neuron_ids
            frames = load_coordinate_frames(path)
            self.assertEqual([frame.frame_id for frame in frames], [5, 7])
            self.assertEqual(frames[1].neuron_ids.tolist(), [30, 42])

    def test_dynamics_h5_bounding_boxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dynamics.h5"
            with h5py.File(path, "w") as handle:
                names = (
                    "ImgStk001_dk001_w1_Dt202510_000099",
                    "ImgStk002_dk001_w1_Dt202510_000000",
                    "ImgStk002_dk001_w1_Dt202510_000001",
                )
                for frame_id, name in enumerate(names):
                    group = handle.create_group(
                        name
                    )
                    points = np.full((3, 8), frame_id, dtype=np.float32)
                    group["d_neuron_pt_tuple_matched_raw_vol"] = points
            frames = load_coordinate_frames(path, source_format="dynamics")
            self.assertEqual([frame.frame_id for frame in frames], [0, 1, 2])
            self.assertEqual(frames[0].neuron_ids.tolist(), [0, 1, 2])
            self.assertEqual(frames[1].name, names[1])
            self.assertEqual(frames[2].coordinates.shape, (3, 8))

    def test_neuron_pt_tuple_npy_bounding_boxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "neuron_pt_tuple.npy"
            np.save(path, np.ones((4, 3, 6), dtype=np.float32))
            frames = load_coordinate_frames(path, source_format="npy")
            self.assertEqual([frame.frame_id for frame in frames], [0, 1, 2, 3])
            self.assertEqual(frames[2].neuron_ids.tolist(), [0, 1, 2])

    def test_realtime_result_directory_like_roi_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for frame_id in (5, 7):
                with h5py.File(root / ("volume_%d.h5" % frame_id), "w") as handle:
                    analysis = handle.create_group("real-time_analysis")
                    group = analysis.create_group("volume_%08d" % frame_id)
                    group["neuron_pt_tuple"] = np.full(
                        (2, 8), frame_id, dtype=np.float32
                    )
                    group["neuron_pred_ids"] = (
                        [3, 9] if frame_id == 5 else [-1, 9]
                    )
            frames = load_coordinate_frames(
                root,
                source_format="realtime-results",
                expected_frame_ids=[5, 6, 7],
            )
            self.assertEqual([frame.frame_id for frame in frames], [5, 6, 7])
            self.assertEqual(frames[0].neuron_ids.tolist(), [3, 9])
            self.assertEqual(frames[1].coordinates.shape, (0, 8))
            self.assertEqual(frames[2].neuron_ids.tolist(), [9])

    def test_tiff_slices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in range(8):
                tifffile.imwrite(
                    root / ("%08d.tif" % index),
                    np.full((12, 10), index, dtype=np.uint16),
                )
            series = open_volume_series(
                root, tiff_layout="slices", frames_per_volume=4, z_range=None
            )
            self.assertEqual(len(series), 2)
            np.testing.assert_array_equal(
                series.frames[1].load()[:, 0, 0], [4, 5, 6, 7]
            )

    def test_sampled_registration_uses_seeded_positions(self):
        loaders = [
            (lambda index=index: np.full((2, 4, 5), index, dtype=np.float32))
            for index in range(20)
        ]

        def fake_ecc(reference_mip, calcium_mip, **kwargs):
            index = int(reference_mip[0, 0])
            matrix = _euclidean_matrix(0.0, index * 0.01, 0.0)
            return matrix, {
                "correlation": 0.9,
                "estimated_matrix": matrix.tolist(),
                "reason": None,
            }

        expected_positions = sorted(
            int(index)
            for index in np.random.default_rng(72).choice(
                20, size=10, replace=False
            )
        )
        with patch(
            "signal_extraction.registration.estimate_registration_matrix",
            side_effect=fake_ecc,
        ):
            _, diagnostics = estimate_sampled_registration_matrix(
                loaders,
                loaders,
                num_samples=10,
                seed=72,
                max_translation_deviation_px=100.0,
            )
        self.assertEqual(diagnostics["sampled_positions"], expected_positions)
        self.assertEqual(diagnostics["num_sampled_volumes"], 10)

    def test_registration_does_not_fallback_for_rotation_deviation(self):
        image = np.arange(20, dtype=np.float32).reshape(4, 5)
        estimated = _euclidean_matrix(8.0, 0.0, 0.0)
        with patch("cv2.findTransformECC", return_value=(0.9, estimated)):
            matrix, diagnostics = estimate_registration_matrix(
                image,
                image,
                flip_horizontal=False,
            )
        np.testing.assert_allclose(matrix, estimated)
        self.assertGreater(diagnostics["rotation_deviation_deg"], 0.3)

    def test_sampled_registration_rejects_outlier_then_takes_median(self):
        matrices = [
            _euclidean_matrix(-0.2, 9.8, -2.2),
            _euclidean_matrix(-0.1, 9.9, -2.1),
            _euclidean_matrix(0.0, 10.0, -2.0),
            _euclidean_matrix(0.1, 10.1, -1.9),
            _euclidean_matrix(0.2, 10.2, -1.8),
            _euclidean_matrix(8.0, 100.0, 60.0),
        ]
        loaders = [
            (lambda index=index: np.full((2, 4, 5), index, dtype=np.float32))
            for index in range(len(matrices))
        ]

        def fake_ecc(reference_mip, calcium_mip, **kwargs):
            matrix = matrices[int(reference_mip[0, 0])]
            return matrix, {
                "correlation": 0.9,
                "estimated_matrix": matrix.tolist(),
                "reason": None,
            }

        with patch(
            "signal_extraction.registration.estimate_registration_matrix",
            side_effect=fake_ecc,
        ):
            matrix, diagnostics = estimate_sampled_registration_matrix(
                loaders,
                loaders,
                num_samples=6,
                min_successful_samples=3,
            )
        np.testing.assert_allclose(
            matrix, _euclidean_matrix(0.0, 10.0, -2.0), atol=1e-6
        )
        self.assertEqual(diagnostics["num_ecc_candidates"], 6)
        self.assertEqual(diagnostics["num_consistent_candidates"], 5)

    def test_sampled_registration_does_not_reject_rotation_deviation(self):
        matrices = [
            _euclidean_matrix(0.0, 10.0, -2.0),
            _euclidean_matrix(0.1, 10.1, -2.1),
            _euclidean_matrix(8.0, 10.2, -1.9),
        ]
        loaders = [
            (lambda index=index: np.full((2, 4, 5), index, dtype=np.float32))
            for index in range(len(matrices))
        ]

        def fake_ecc(reference_mip, calcium_mip, **kwargs):
            matrix = matrices[int(reference_mip[0, 0])]
            return matrix, {
                "correlation": 0.9,
                "estimated_matrix": matrix.tolist(),
                "reason": None,
            }

        with patch(
            "signal_extraction.registration.estimate_registration_matrix",
            side_effect=fake_ecc,
        ):
            _, diagnostics = estimate_sampled_registration_matrix(
                loaders,
                loaders,
                num_samples=3,
                min_successful_samples=3,
            )
        self.assertEqual(diagnostics["num_consistent_candidates"], 3)
        self.assertGreater(
            diagnostics["samples"][2]["rotation_deviation_deg"],
            0.3,
        )

    def test_sampled_registration_requires_enough_candidates(self):
        loaders = [lambda: np.ones((2, 4, 5), dtype=np.float32) for _ in range(4)]
        matrix = _euclidean_matrix(0.0, 0.0, 0.0)
        diagnostics = {
            "correlation": 0.0,
            "estimated_matrix": matrix.tolist(),
            "reason": None,
        }
        with patch(
            "signal_extraction.registration.estimate_registration_matrix",
            return_value=(matrix, diagnostics),
        ):
            with self.assertRaises(RegistrationCalibrationError) as context:
                estimate_sampled_registration_matrix(loaders, loaders)
        self.assertEqual(context.exception.diagnostics["num_ecc_candidates"], 0)

    def test_end_to_end_npy(self):
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference"
            calcium = root / "calcium"
            reference.mkdir()
            calcium.mkdir()
            coordinates = []
            for time_index in range(3):
                volume = rng.integers(
                    100, 1000, size=(18, 48, 52), dtype=np.uint16
                )
                np.save(reference / ("%06d.npy" % time_index), volume)
                np.save(
                    calcium / ("%06d.npy" % time_index),
                    np.flip(volume, axis=2).copy(),
                )
                coordinates.append(
                    [[18, 22, 40, 10, 12, 20], [35, 30, 55, 8, 8, 15]]
                )
            coordinate_path = root / "coords.npy"
            np.save(coordinate_path, np.asarray(coordinates, dtype=np.float32))
            output_path = root / "intensity.h5"
            result = extract_signals(
                coordinate_path,
                reference,
                calcium,
                output_path,
                device="cpu",
                register_calcium=False,
                z_range=None,
                show_progress=False,
            )
            self.assertEqual(result.activity.shape, (2, 2, 3))
            self.assertTrue(np.isfinite(result.activity).all())
            with h5py.File(output_path, "r") as handle:
                self.assertEqual(handle["activity"].shape, (2, 2, 3))
                self.assertIn("parameters_json", handle.attrs)

    def test_pipeline_defaults_to_sampled_registration(self):
        rng = np.random.default_rng(12)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference"
            calcium = root / "calcium"
            reference.mkdir()
            calcium.mkdir()
            coordinates = []
            for time_index in range(3):
                volume = rng.integers(
                    100, 1000, size=(6, 24, 28), dtype=np.uint16
                )
                np.save(reference / ("%06d.npy" % time_index), volume)
                np.save(
                    calcium / ("%06d.npy" % time_index),
                    np.flip(volume, axis=2).copy(),
                )
                coordinates.append([[14, 12, 15, 6, 6, 10]])
            coordinate_path = root / "coords.npy"
            np.save(coordinate_path, np.asarray(coordinates, dtype=np.float32))
            identity = np.eye(2, 3, dtype=np.float32)
            diagnostics = {
                "strategy": "sample_median",
                "matrix": identity.tolist(),
            }
            with patch(
                "signal_extraction.pipeline."
                "estimate_sampled_registration_matrix",
                return_value=(identity, diagnostics),
            ) as sampled:
                result = extract_signals(
                    coordinate_path,
                    reference,
                    calcium,
                    device="cpu",
                    z_range=None,
                    show_progress=False,
                )
            self.assertEqual(sampled.call_count, 1)
            self.assertEqual(sampled.call_args.kwargs["num_samples"], 10)
            self.assertEqual(sampled.call_args.kwargs["seed"], 72)
            self.assertEqual(sampled.call_args.kwargs["frame_ids"], [0, 1, 2])
            self.assertEqual(
                result.registration_diagnostics["source"],
                "sampled_ecc_median",
            )


if __name__ == "__main__":
    unittest.main()

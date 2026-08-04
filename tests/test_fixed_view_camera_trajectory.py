from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plot_fixed_view_camera_trajectory.py"
SPEC = importlib.util.spec_from_file_location("fixed_view", SCRIPT)
fixed_view = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = fixed_view
SPEC.loader.exec_module(fixed_view)


class FixedViewTrajectoryTests(unittest.TestCase):
    def test_requested_combined_title_is_exact(self) -> None:
        self.assertEqual(
            fixed_view.COMBINED_TITLE,
            "Ground Truth DoF vs Predicted DoF",
        )

    def test_two_distinct_fixed_views_are_defined(self) -> None:
        self.assertEqual(len(fixed_view.DEFAULT_VIEWS), 2)
        self.assertNotEqual(
            fixed_view.DEFAULT_VIEWS[0].azimuth_degrees,
            fixed_view.DEFAULT_VIEWS[1].azimuth_degrees,
        )

    def test_trim_trace_drops_spawn_rows(self) -> None:
        rows = [
            {"location_cm": np.asarray((-200.0, 0.0, 30.0)), "rotation_rpy": np.zeros(3), "gaze_confidence": 0.0},
            {"location_cm": np.asarray((10.0, 20.0, 160.0)), "rotation_rpy": np.zeros(3), "gaze_confidence": 0.8},
        ]
        self.assertEqual(fixed_view.trim_trace(rows, "tracked-gaze"), rows[1:])

    def test_resample_interval_interpolates_xyz(self) -> None:
        rows = [
            {"timestamp_s": 2.0, "location_cm": np.asarray((0.0, 0.0, 0.0))},
            {"timestamp_s": 4.0, "location_cm": np.asarray((2.0, 4.0, 6.0))},
        ]
        times, xyz = fixed_view.resample_interval(rows, 3)
        np.testing.assert_allclose(times, [2.0, 3.0, 4.0])
        np.testing.assert_allclose(xyz[1], [1.0, 2.0, 3.0])

    def test_equal_axis_limits_use_common_span_and_include_prediction(self) -> None:
        scene = np.asarray([[-10.0, -10.0, -10.0], [10.0, 10.0, 10.0]])
        gt = np.asarray([[0.0, 0.0, 0.0], [10.0, 20.0, 30.0]])
        predicted = np.asarray([[250.0, -300.0, 175.0]])
        limits = fixed_view.equal_axis_limits(
            scene, [gt, predicted], padding_fraction=0.0
        )
        spans = [high - low for low, high in limits]
        np.testing.assert_allclose(spans, [spans[0]] * 3)
        for coordinate, (low, high) in zip(predicted[0], limits):
            self.assertLessEqual(low, coordinate)
            self.assertGreaterEqual(high, coordinate)

    def test_dancenet_points_transform_to_trace_coordinates(self) -> None:
        source_m = np.asarray([[1.0, 2.0, 3.0]])
        actual_cm = (source_m @ fixed_view.EVOGS_TO_GSV.T) * 100.0
        np.testing.assert_allclose(actual_cm, [[300.0, 100.0, -200.0]])

    def test_position_error_summary(self) -> None:
        gt = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        predicted = np.asarray([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])
        summary = fixed_view.position_error_summary(gt, predicted)
        self.assertAlmostEqual(summary["mean_cm"], 2.5)
        self.assertAlmostEqual(summary["rmse_cm"], np.sqrt(12.5))
        self.assertAlmostEqual(summary["max_cm"], 5.0)


if __name__ == "__main__":
    unittest.main()

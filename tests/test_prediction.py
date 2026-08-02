import csv
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path


NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(NUMPY_AVAILABLE, "prediction extra requires numpy")
class LinearPredictionTests(unittest.TestCase):
    def test_recall_constrained_threshold_uses_highest_feasible_value(self) -> None:
        import numpy as np
        from fovsim.prediction import _select_recall_constrained_threshold

        prediction = np.asarray([[0.9, 0.8, 0.3, 0.2]])
        target = np.asarray([[1.0, 1.0, 0.0, 0.0]])
        threshold, metrics, constraint_met = (
            _select_recall_constrained_threshold(
                prediction,
                target,
                target_threshold=0.5,
                minimum=0.1,
                maximum=0.9,
                steps=9,
                minimum_recall=1.0,
            )
        )
        self.assertTrue(constraint_met)
        self.assertAlmostEqual(threshold, 0.8)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)

    @staticmethod
    def _write_trace(path: Path, trace_index: int) -> None:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "FileName",
                    "LocationX",
                    "LocationY",
                    "LocationZ",
                    "RotationRoll",
                    "RotationPitch",
                    "RotationYaw",
                    "GazeHitX",
                    "GazeHitY",
                    "GazeHitZ",
                    "GazeConfidence",
                    "Frame",
                    "Timestamp",
                ]
            )
            for frame in range(31):
                time_s = frame / 10.0
                yaw_radians = math.radians(179.0 + 4.0 * time_s)
                writer.writerow(
                    [
                        "BiancaGolden_CircleTurns",
                        trace_index + 2.0 * time_s,
                        -trace_index + time_s,
                        30.0 + 0.5 * time_s,
                        0.1 * time_s,
                        2.0 * time_s,
                        179.0 + 4.0 * time_s,
                        math.cos(yaw_radians),
                        math.sin(yaw_radians),
                        0.0,
                        0.998,
                        frame,
                        time_s,
                    ]
                )

    @staticmethod
    def _write_visibility(path: Path, trace_index: int) -> None:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "output_frame",
                    "trace_timestamp_s",
                    "cell_id",
                    "contributing_gaussian_fraction",
                ]
            )
            for frame in range(31):
                time_s = frame / 10.0
                writer.writerow(
                    [frame, time_s, "0:0:0", min(1.0, 0.02 * frame)]
                )
                writer.writerow(
                    [
                        frame,
                        time_s,
                        "1:0:0",
                        max(0.0, 1.0 - 0.02 * frame - 0.01 * trace_index),
                    ]
                )

    def test_trace_split_dof_to_visibility_and_outputs(self) -> None:
        from fovsim.prediction import run_linear_prediction

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            traces = root / "traces"
            visibility = root / "visibility"
            output = root / "output"
            traces.mkdir()
            visibility.mkdir()
            for trace_index in range(5):
                name = f"trace_{trace_index}"
                self._write_trace(traces / f"{name}.csv", trace_index)
                self._write_visibility(
                    visibility / f"{name}.csv", trace_index
                )

            summary = run_linear_prediction(
                trace_dir=traces,
                visibility_dir=visibility,
                output_dir=output,
                fps=10.0,
                history_ms=500,
                horizons_ms=(100, 200),
                target_mode="binary",
                feature_mode="raw_gaze",
                seed=7,
                expected_traces=5,
                safe_recall_target=0.85,
            )

            self.assertEqual(summary["status"], "PASS")
            self.assertEqual(len(summary["split"]["training_traces"]), 4)
            self.assertEqual(len(summary["split"]["test_traces"]), 1)
            self.assertEqual(summary["cell_count"], 2)
            self.assertEqual(summary["config"]["training_target_mode"], "binary")
            self.assertEqual(summary["config"]["feature_mode"], "raw_gaze")
            selected_threshold = summary["horizons"]["200ms"][
                "threshold_calibration"
            ]["selected_decision_threshold"]
            self.assertGreaterEqual(selected_threshold, 0.01)
            self.assertLessEqual(selected_threshold, 0.5)
            safe = summary["horizons"]["200ms"][
                "safe_threshold_calibration"
            ]
            self.assertEqual(safe["minimum_recall"], 0.85)
            self.assertGreaterEqual(
                safe["selected_decision_threshold"], 0.01
            )
            self.assertLessEqual(
                safe["selected_decision_threshold"], 0.5
            )
            self.assertIn("100ms", summary["horizons"])
            self.assertGreaterEqual(
                summary["horizons"]["200ms"]["visibility"][
                    "classification"
                ]["accuracy"],
                0.0,
            )
            classification = summary["horizons"]["200ms"]["visibility"][
                "classification"
            ]
            self.assertGreaterEqual(classification["f2"], 0.0)
            self.assertGreaterEqual(
                classification["missed_visible_cells_per_frame"], 0.0
            )
            self.assertGreaterEqual(classification["extra_cells_per_frame"], 0.0)
            self.assertTrue(
                (output / "visibility_model_200ms.npz").is_file()
            )
            self.assertTrue((output / "per_trace_metrics.csv").is_file())
            saved = json.loads(
                (output / "linear_visibility_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(saved["config"]["history_steps"], 5)

    def test_saved_model_generates_qoe_policy_without_future_labels(self) -> None:
        import numpy as np
        from fovsim.predicted_policy import generate_predicted_policy
        from fovsim.prediction import StandardizedRidge

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace = root / "trace.csv"
            self._write_trace(trace, 0)
            visibility_paths = []
            for variant in (0, 1):
                path = root / f"visibility_{variant}.csv"
                visibility_paths.append(path)
                with path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow([
                        "output_frame", "output_time_s", "trace_source_row",
                        "trace_timestamp_s", "gsv_frame", "cell_id",
                        "contributing_gaussian_fraction",
                    ])
                    for frame in range(31):
                        for cell_id in ("0:0:0", "1:0:0"):
                            writer.writerow([
                                frame, frame / 10, frame + 2, frame / 10,
                                frame, cell_id, float(variant),
                            ])
            model = StandardizedRidge(
                feature_mean=np.zeros(36), feature_scale=np.ones(36),
                target_mean=np.asarray([0.8, 0.1]), target_scale=np.ones(2),
                coefficients=np.zeros((36, 2)), alpha=1.0,
            )
            model_path = root / "model.npz"
            model.save(
                model_path, cell_ids=("0:0:0", "1:0:0"),
                decision_threshold=0.5, target_threshold=0.5,
                training_target_mode="binary",
            )
            outputs = []
            for variant, visibility in enumerate(visibility_paths):
                output = root / f"output_{variant}"
                summary = generate_predicted_policy(
                    trace_path=trace, visibility_path=visibility,
                    model_path=model_path, output_dir=output, fps=10,
                    history_ms=500, horizon_ms=500,
                )
                self.assertFalse(summary["future_visibility_values_used"])
                outputs.append((output / "cell_decisions.csv").read_text())
            self.assertEqual(outputs[0], outputs[1])
            rows = list(csv.DictReader(outputs[0].splitlines()))
            self.assertEqual({row["target_level"] for row in rows}, {"0", "3"})
            self.assertEqual([int(row["output_frame"]) for row in rows[:2]], [0, 0])

    def test_motion_quadratic_model_and_policy_guard_band(self) -> None:
        import numpy as np
        from fovsim.predicted_policy import generate_predicted_policy
        from fovsim.prediction import StandardizedRidge, _history_features

        histories = np.arange(4 * 6 * 6, dtype=float).reshape(4, 6, 6)
        features = _history_features(
            histories, fps=10.0, horizon_s=0.5,
            feature_mode="motion_quadratic",
        )
        self.assertEqual(features.shape, (4, 87))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace = root / "trace.csv"
            visibility = root / "visibility.csv"
            self._write_trace(trace, 0)
            with visibility.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow([
                    "output_frame", "output_time_s", "trace_source_row",
                    "trace_timestamp_s", "gsv_frame", "cell_id",
                    "contributing_gaussian_fraction",
                ])
                for frame in range(31):
                    writer.writerow([frame, frame / 10, frame + 2, frame / 10,
                                     frame, "0:0:0", 1.0])
                    writer.writerow([frame, frame / 10, frame + 2, frame / 10,
                                     frame, "1:0:0", 0.0])
            model = StandardizedRidge(
                feature_mean=np.zeros(87), feature_scale=np.ones(87),
                target_mean=np.asarray([0.8, 0.1]), target_scale=np.ones(2),
                coefficients=np.zeros((87, 2)), alpha=1.0,
            )
            model_path = root / "model.npz"
            model.save(
                model_path, cell_ids=("0:0:0", "1:0:0"),
                decision_threshold=0.5, target_threshold=0.5,
                training_target_mode="binary", feature_mode="motion_quadratic",
            )
            guarded = root / "guarded"
            result = generate_predicted_policy(
                trace_path=trace, visibility_path=visibility,
                model_path=model_path, output_dir=guarded, fps=10,
                history_ms=500, horizon_ms=500, guard_band_steps=1,
            )
            self.assertEqual(result["feature_mode"], "motion_quadratic")
            self.assertEqual(result["mean_selected_cells"], 2.0)
            base = generate_predicted_policy(
                trace_path=trace, visibility_path=visibility,
                model_path=model_path, output_dir=root / "base", fps=10,
                history_ms=500, horizon_ms=500, policy_mode="base_only",
            )
            self.assertEqual(base["mean_selected_cells"], 0.0)
            persistence = generate_predicted_policy(
                trace_path=trace, visibility_path=visibility,
                model_path=model_path, output_dir=root / "persistence", fps=10,
                history_ms=500, horizon_ms=500, policy_mode="persistence",
            )
            self.assertEqual(persistence["mean_selected_cells"], 1.0)

    def test_motion_gaze_features_and_saved_policy(self) -> None:
        import numpy as np
        from fovsim.predicted_policy import generate_predicted_policy
        from fovsim.prediction import StandardizedRidge, _history_features

        histories = np.arange(4 * 6 * 6, dtype=float).reshape(4, 6, 6)
        gaze = np.zeros((4, 6, 3), dtype=float)
        gaze[:, :, 0] = 1.0
        features = _history_features(
            histories, fps=10.0, horizon_s=0.5,
            feature_mode="motion_gaze", gaze_histories=gaze,
        )
        self.assertEqual(features.shape, (4, 135))
        raw_gaze_features = _history_features(
            histories, fps=10.0, horizon_s=0.5,
            feature_mode="raw_gaze", gaze_histories=gaze,
        )
        self.assertEqual(raw_gaze_features.shape, (4, 84))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace = root / "trace.csv"
            visibility = root / "visibility.csv"
            self._write_trace(trace, 0)
            with visibility.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow([
                    "output_frame", "output_time_s", "trace_source_row",
                    "trace_timestamp_s", "gsv_frame", "cell_id",
                    "contributing_gaussian_fraction",
                ])
                for frame in range(31):
                    writer.writerow([
                        frame, frame / 10, frame + 2, frame / 10,
                        frame, "0:0:0", 1.0,
                    ])
            model = StandardizedRidge(
                feature_mean=np.zeros(84), feature_scale=np.ones(84),
                target_mean=np.asarray([0.8]), target_scale=np.ones(1),
                coefficients=np.zeros((84, 1)), alpha=1.0,
            )
            model_path = root / "gaze_model.npz"
            model.save(
                model_path, cell_ids=("0:0:0",), decision_threshold=0.5,
                target_threshold=0.5, training_target_mode="binary",
                feature_mode="raw_gaze",
            )
            result = generate_predicted_policy(
                trace_path=trace, visibility_path=visibility,
                model_path=model_path, output_dir=root / "policy", fps=10,
                history_ms=500, horizon_ms=500,
            )
            self.assertEqual(result["feature_mode"], "raw_gaze")
            self.assertEqual(result["input_contract"], "6dof_and_gaze_history")
            self.assertEqual(result["mean_selected_cells"], 1.0)

    def test_current_visibility_linear_feature_is_causal(self) -> None:
        import numpy as np
        from fovsim.predicted_policy import generate_predicted_policy
        from fovsim.prediction import StandardizedRidge

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace = root / "trace.csv"
            visibility = root / "visibility.csv"
            self._write_trace(trace, 0)
            fractions = {}
            with visibility.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow([
                    "output_frame", "output_time_s", "trace_source_row",
                    "trace_timestamp_s", "gsv_frame", "cell_id",
                    "contributing_gaussian_fraction",
                ])
                for frame in range(31):
                    for cell_index, cell_id in enumerate(("0:0:0", "1:0:0")):
                        fraction = (frame + cell_index) % 2
                        fractions[(frame, cell_id)] = float(fraction)
                        writer.writerow([
                            frame, frame / 10, frame + 2, frame / 10,
                            frame, cell_id, fraction,
                        ])
            feature_count = 84 + 2
            coefficients = np.zeros((feature_count, 2))
            coefficients[-2:, :] = np.eye(2)
            model = StandardizedRidge(
                feature_mean=np.zeros(feature_count),
                feature_scale=np.ones(feature_count),
                target_mean=np.zeros(2), target_scale=np.ones(2),
                coefficients=coefficients, alpha=1.0,
            )
            model_path = root / "model.npz"
            model.save(
                model_path, cell_ids=("0:0:0", "1:0:0"),
                decision_threshold=0.5, target_threshold=0.5,
                training_target_mode="binary",
                feature_mode="raw_gaze_current_visibility",
            )
            result = generate_predicted_policy(
                trace_path=trace, visibility_path=visibility,
                model_path=model_path, output_dir=root / "policy",
                fps=10, history_ms=500, horizon_ms=100,
            )
            self.assertTrue(result["current_visibility_values_used"])
            self.assertFalse(result["future_visibility_values_used"])
            rows = list(csv.DictReader(
                (root / "policy" / "cell_decisions.csv").read_text(
                    encoding="utf-8"
                ).splitlines()
            ))
            for row in rows:
                current_frame = int(row["prediction_current_output_frame"])
                expected = fractions[(current_frame, row["cell_id"])]
                self.assertEqual(
                    float(row["predicted_visibility_score"]), expected
                )


if __name__ == "__main__":
    unittest.main()

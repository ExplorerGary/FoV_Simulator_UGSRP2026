import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(NUMPY_AVAILABLE, "prediction extra requires numpy")
class LinearPredictionTests(unittest.TestCase):
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
                    "Frame",
                    "Timestamp",
                ]
            )
            for frame in range(31):
                time_s = frame / 10.0
                writer.writerow(
                    [
                        "BiancaGolden_CircleTurns",
                        trace_index + 2.0 * time_s,
                        -trace_index + time_s,
                        30.0 + 0.5 * time_s,
                        0.1 * time_s,
                        2.0 * time_s,
                        179.0 + 4.0 * time_s,
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
                seed=7,
                expected_traces=5,
            )

            self.assertEqual(summary["status"], "PASS")
            self.assertEqual(len(summary["split"]["training_traces"]), 4)
            self.assertEqual(len(summary["split"]["test_traces"]), 1)
            self.assertEqual(summary["cell_count"], 2)
            self.assertEqual(summary["config"]["training_target_mode"], "binary")
            selected_threshold = summary["horizons"]["200ms"][
                "threshold_calibration"
            ]["selected_decision_threshold"]
            self.assertGreaterEqual(selected_threshold, 0.01)
            self.assertLessEqual(selected_threshold, 0.5)
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


if __name__ == "__main__":
    unittest.main()

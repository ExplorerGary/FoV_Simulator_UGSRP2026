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


if __name__ == "__main__":
    unittest.main()

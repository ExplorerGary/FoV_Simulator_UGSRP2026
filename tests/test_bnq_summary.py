import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_lr500_bnq.py"


class BnqSummaryTests(unittest.TestCase):
    @staticmethod
    def load_module():
        specification = importlib.util.spec_from_file_location(
            "summarize_lr500_bnq", SCRIPT
        )
        assert specification is not None and specification.loader is not None
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        return module

    def test_gt_bytes_uses_fallback_and_reports_true_missing_assets(self) -> None:
        module = self.load_module()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            decisions = root / "decisions.csv"
            with decisions.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["output_frame", "asset_frame_id"])
                writer.writerow([0, 41])
                writer.writerow([0, 41])
                writer.writerow([1, 42])
                writer.writerow([2, 43])
            first = root / "parent"
            second = first / "BiancaGolden_CircleTurns"
            second.mkdir(parents=True)
            (second / "0000041.ply").write_bytes(b"a" * 7)
            (second / "0000042.ply").write_bytes(b"b" * 11)

            total, missing = module.gt_bytes(decisions, [first, second])

            self.assertEqual(total, 18)
            self.assertEqual(missing, {43})

    def test_cell_metrics_align_on_source_frame_after_skipped_frame(self) -> None:
        module = self.load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            visibility = root / "visibility.csv"
            decisions = root / "decisions.csv"
            with visibility.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    [
                        "output_frame", "source_output_frame", "cell_id",
                        "contributing_gaussian_fraction",
                    ]
                )
                writer.writerow([0, 0, "0:0:0", 0.8])
                # source frame 1 was skipped; emitted output frame is compacted.
                writer.writerow([1, 2, "0:0:0", 0.9])
            with decisions.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    [
                        "output_frame", "source_output_frame", "cell_id",
                        "target_level", "predicted_visibility_score",
                    ]
                )
                writer.writerow([0, 0, "0:0:0", 3, 0.8])
                writer.writerow([1, 2, "0:0:0", 3, 0.9])

            metrics = module.policy_cell_metrics(decisions, visibility)

            self.assertEqual(metrics["samples"], 2)
            self.assertEqual(metrics["true_positive"], 2)
            self.assertEqual(metrics["false_positive"], 0)
            self.assertEqual(metrics["false_negative"], 0)
            self.assertEqual(metrics["squared_error"], 0.0)
            self.assertEqual(metrics["precision"], 1.0)
            self.assertEqual(metrics["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()

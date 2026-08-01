import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_lr500_bnq.py"


class BnqSummaryTests(unittest.TestCase):
    def test_gt_bytes_uses_fallback_and_reports_true_missing_assets(self) -> None:
        specification = importlib.util.spec_from_file_location(
            "summarize_lr500_bnq", SCRIPT
        )
        assert specification is not None and specification.loader is not None
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)

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


if __name__ == "__main__":
    unittest.main()

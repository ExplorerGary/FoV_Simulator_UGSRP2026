import csv
import json
import tempfile
import unittest
from pathlib import Path

from fovsim.pipeline import run_simulation


EXAMPLES = Path(__file__).parents[1] / "examples"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


class PipelineTests(unittest.TestCase):
    def test_run_simulation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            result = run_simulation(
                trace_path=EXAMPLES / "trace.csv",
                visibility_path=EXAMPLES / "visibility.csv",
                output_dir=output,
                threshold=0.5,
                first_frame=1,
                frame_count=3,
            )

            self.assertEqual(result.frame_count, 3)
            self.assertEqual(result.decision_rows, 6)
            self.assertAlmostEqual(result.mean_enhancement3_cells, 1.0)
            self.assertAlmostEqual(
                result.mean_enhancement3_cell_fraction,
                0.5,
            )

            decisions = read_csv(output / "cell_decisions.csv")
            self.assertEqual(
                [int(row["target_level"]) for row in decisions],
                [3, 0, 3, 0, 3, 0],
            )
            self.assertEqual(
                [row["enhancement_required"] for row in decisions],
                ["1", "0", "1", "0", "1", "0"],
            )

            frames = read_csv(output / "frame_summary.csv")
            self.assertEqual(len(frames), 3)
            self.assertEqual(
                [int(row["enhancement3_cells"]) for row in frames],
                [1, 1, 1],
            )

            metadata = json.loads(
                (output / "run_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["policy"]["comparison"], ">=")
            self.assertEqual(metadata["policy"]["threshold"], 0.5)
            self.assertEqual(metadata["trace_summary"]["row_count"], 3)

    def test_trace_visibility_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            visibility = temporary_path / "mismatch.csv"
            contents = (EXAMPLES / "visibility.csv").read_text(
                encoding="utf-8"
            )
            visibility.write_text(
                contents.replace(",2,0.000000,0,", ",99,0.000000,0,", 1),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "trace source row 99"):
                run_simulation(
                    trace_path=EXAMPLES / "trace.csv",
                    visibility_path=visibility,
                    output_dir=temporary_path / "out",
                )


if __name__ == "__main__":
    unittest.main()

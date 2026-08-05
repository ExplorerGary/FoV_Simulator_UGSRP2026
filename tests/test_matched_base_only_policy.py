import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "generate_matched_base_only_policy.py"
)


class MatchedBaseOnlyPolicyTests(unittest.TestCase):
    @staticmethod
    def load_module():
        specification = importlib.util.spec_from_file_location(
            "generate_matched_base_only_policy", SCRIPT
        )
        assert specification is not None and specification.loader is not None
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        return module

    def test_preserves_alignment_and_disables_every_enhancement(self) -> None:
        module = self.load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.csv"
            with source.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "output_frame", "source_output_frame", "asset_frame_id",
                        "cell_id", "predicted_visibility_score",
                        "enhancement_required", "target_level",
                    ],
                )
                writer.writeheader()
                writer.writerows([
                    {
                        "output_frame": 0, "source_output_frame": 9,
                        "asset_frame_id": 10, "cell_id": "0:0:0",
                        "predicted_visibility_score": 0.8,
                        "enhancement_required": 1, "target_level": 3,
                    },
                    {
                        "output_frame": 1, "source_output_frame": 12,
                        "asset_frame_id": 13, "cell_id": "1:0:0",
                        "predicted_visibility_score": 0.1,
                        "enhancement_required": 0, "target_level": 0,
                    },
                ])

            summary = module.generate(source, root / "output", "matched_base_only")
            with (root / "output" / "cell_decisions.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))

            self.assertEqual(summary["frame_count"], 2)
            self.assertEqual(summary["mean_selected_cells"], 0.0)
            self.assertEqual([row["source_output_frame"] for row in rows], ["9", "12"])
            self.assertEqual([row["asset_frame_id"] for row in rows], ["10", "13"])
            self.assertTrue(all(row["enhancement_required"] == "0" for row in rows))
            self.assertTrue(all(row["target_level"] == "0" for row in rows))
            self.assertTrue(all(row["predicted_visibility_score"] == "0.0" for row in rows))


if __name__ == "__main__":
    unittest.main()

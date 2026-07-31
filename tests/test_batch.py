import json
import tempfile
import unittest
from pathlib import Path

from fovsim.batch import load_manifest, run_batch


EXAMPLES = Path(__file__).parents[1] / "examples"


class BatchTests(unittest.TestCase):
    def test_batch_manifest_and_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            manifest = temporary_path / "jobs.jsonl"
            jobs = [
                {
                    "name": "job_a",
                    "trace": str(EXAMPLES / "trace.csv"),
                    "visibility": str(EXAMPLES / "visibility.csv"),
                    "output_dir": str(temporary_path / "job_a"),
                    "threshold": 0.5,
                    "frame_count": 3,
                },
                {
                    "name": "job_b",
                    "trace": str(EXAMPLES / "trace.csv"),
                    "visibility": str(EXAMPLES / "visibility.csv"),
                    "output_dir": str(temporary_path / "job_b"),
                    "threshold": 0.75,
                    "frame_count": 3,
                },
            ]
            manifest.write_text(
                "\n".join(json.dumps(job) for job in jobs) + "\n",
                encoding="utf-8",
            )

            loaded = load_manifest(manifest)
            self.assertEqual([job.name for job in loaded], ["job_a", "job_b"])
            results = run_batch(
                manifest_path=manifest,
                workers=1,
                summary_dir=temporary_path / "summary",
            )

            self.assertEqual(
                [result["name"] for result in results],
                ["job_a", "job_b"],
            )
            self.assertTrue(
                all(result["status"] == "PASS" for result in results)
            )
            self.assertTrue(
                (temporary_path / "summary" / "batch_summary.csv").is_file()
            )
            self.assertTrue(
                (temporary_path / "summary" / "batch_summary.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

from fovsim.trace import load_trace


EXAMPLES = Path(__file__).parents[1] / "examples"


class TraceTests(unittest.TestCase):
    def test_load_trace_summary(self) -> None:
        rows, summary = load_trace(EXAMPLES / "trace.csv")

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0].source_row, 2)
        self.assertEqual(rows[0].location_cm, (-200.0, 0.0, 30.0))
        self.assertEqual(rows[-1].rotation_rpy_degrees, (0.2, 2.0, 4.0))
        self.assertEqual(summary.row_count, 3)
        self.assertEqual(summary.first_gsv_frame, 0)
        self.assertEqual(summary.last_gsv_frame, 2)
        self.assertAlmostEqual(summary.effective_fps, 90.0009, places=3)

    def test_repeated_header_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trace = Path(temporary) / "bad.csv"
            trace.write_text(
                "FileName,LocationX,LocationY,LocationZ,RotationRoll,"
                "RotationPitch,RotationYaw,Frame,Timestamp\n"
                "x.gsv,0,0,0,0,0,0,0,0\n"
                "FileName,LocationX,LocationY,LocationZ,RotationRoll,"
                "RotationPitch,RotationYaw,Frame,Timestamp\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Repeated CSV header"):
                load_trace(trace)

    def test_optional_gaze_direction_is_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trace = Path(temporary) / "gaze.csv"
            trace.write_text(
                "FileName,LocationX,LocationY,LocationZ,RotationRoll,"
                "RotationPitch,RotationYaw,GazeHitX,GazeHitY,GazeHitZ,"
                "GazeConfidence,Frame,Timestamp\n"
                "x.gsv,0,0,0,0,0,0,1,0,0,0.998,0,0\n"
                "x.gsv,0,0,0,0,0,0,1,0,0,0.998,1,1\n",
                encoding="utf-8",
            )
            rows, _ = load_trace(trace)
            self.assertEqual(rows[0].gaze_direction, (1.0, 0.0, 0.0))
            self.assertEqual(rows[0].gaze_confidence, 0.998)


if __name__ == "__main__":
    unittest.main()

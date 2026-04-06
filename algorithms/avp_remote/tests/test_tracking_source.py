import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from avp_snapshot_io import save_snapshot_payload
from tracking_source import TrackingStream


def _sample_payload():
    return {
        "head": [
            [1.0, 0.0, 0.0, 0.25],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.75],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "left_arm": None,
        "right_arm": None,
        "left_wrist": None,
        "right_wrist": None,
        "left_pinch_distance": 0.01,
        "right_pinch_distance": 0.02,
        "left_wrist_roll": 0.11,
        "right_wrist_roll": 0.22,
    }


class TestTrackingSource(unittest.TestCase):
    def test_snapshot_mode_does_not_import_avp_stream(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            snapshot_path = Path(tmp_dir) / "snapshot.json"
            save_snapshot_payload(_sample_payload(), snapshot_path)

            with mock.patch("tracking_source.importlib.import_module", side_effect=AssertionError("unexpected import")):
                stream = TrackingStream(tracking_source="snapshot", snapshot_path=snapshot_path)

            frame = stream.get_tracking_frame()
            self.assertIsNotNone(frame)
            self.assertIsNotNone(frame["head"])
            self.assertIsNone(frame["left_arm"])
            self.assertIsNone(frame["right_arm"])


if __name__ == "__main__":
    unittest.main()

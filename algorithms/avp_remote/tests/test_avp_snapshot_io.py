import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from avp_snapshot_io import (
    SnapshotDecodeError,
    SnapshotNotFoundError,
    load_snapshot_payload,
    save_snapshot_payload,
)


def _sample_payload(head_offset):
    return {
        "head": [
            [1.0, 0.0, 0.0, head_offset],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
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


class TestAvpSnapshotIO(unittest.TestCase):
    def test_roundtrip_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            snapshot_path = Path(tmp_dir) / "snapshot.json"
            payload = _sample_payload(head_offset=0.5)

            save_snapshot_payload(payload, snapshot_path)
            loaded = load_snapshot_payload(snapshot_path)
            self.assertEqual(loaded, payload)

    def test_repeated_save_overwrites_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            snapshot_path = Path(tmp_dir) / "snapshot.json"
            first = _sample_payload(head_offset=0.5)
            second = _sample_payload(head_offset=1.5)

            save_snapshot_payload(first, snapshot_path)
            save_snapshot_payload(second, snapshot_path)
            loaded = load_snapshot_payload(snapshot_path)
            self.assertEqual(loaded, second)

    def test_missing_snapshot_raises_not_found(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            snapshot_path = Path(tmp_dir) / "missing.json"
            with self.assertRaises(SnapshotNotFoundError):
                load_snapshot_payload(snapshot_path)

    def test_invalid_json_raises_decode_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            snapshot_path = Path(tmp_dir) / "snapshot.json"
            snapshot_path.write_text("{not-json", encoding="utf-8")
            with self.assertRaises(SnapshotDecodeError):
                load_snapshot_payload(snapshot_path)

    def test_non_object_json_raises_decode_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            snapshot_path = Path(tmp_dir) / "snapshot.json"
            snapshot_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
            with self.assertRaises(SnapshotDecodeError):
                load_snapshot_payload(snapshot_path)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]


class PlayParallelAnimationImportTests(unittest.TestCase):
    def test_entrypoint_import_is_runtime_clean(self) -> None:
        script = f"""
import sys
sys.path.insert(0, {str(MODULE_ROOT)!r})
import play_parallel_animation  # noqa: F401
print('numpy_loaded', 'numpy' in sys.modules)
print('isaacsim_loaded', 'isaacsim' in sys.modules)
"""
        result = subprocess.run(
            [sys.executable, '-c', script],
            capture_output=True,
            check=True,
            text=True,
        )

        self.assertIn('numpy_loaded False', result.stdout)
        self.assertIn('isaacsim_loaded False', result.stdout)


if __name__ == '__main__':
    unittest.main()

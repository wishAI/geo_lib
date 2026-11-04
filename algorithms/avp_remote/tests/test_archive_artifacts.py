import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('archive_artifacts', Path(__file__).parents[1] / 'archive_artifacts.py')
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


class ArchiveTests(unittest.TestCase):
    def test_verified_archive_is_repeatable_and_keeps_live_recordings(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo, cloud, live = root / 'repo', root / 'cloud', root / 'live'
            repo.mkdir(); live.mkdir(); cloud.mkdir()
            manifest = repo / 'large_files.json'
            manifest.write_text(json.dumps({'cloudRoot': str(cloud), 'files': []}))
            take = live / 'take.json.gz'
            take.write_bytes(b'immutable recording')
            scene = repo / archive.OUTPUTS / 'web_scene/scene.json'
            scene.parent.mkdir(parents=True)
            scene.write_bytes(b'generated scene')

            def rclone(command, **kwargs):
                if command[1] == 'copy':
                    shutil.copytree(command[2], cloud, dirs_exist_ok=True)
                elif command[1] == 'check':
                    for p in Path(command[2]).rglob('*'):
                        if p.is_file():
                            self.assertEqual(p.read_bytes(), (cloud / 'assets/avp_remote' / p.relative_to(command[2])).read_bytes())
                return subprocess.CompletedProcess(command, 0)

            with patch.object(archive, 'REPO', repo), patch.dict(archive.os.environ, {'GEO_CLOUD_ROOT': str(cloud)}):
                before = manifest.read_bytes()
                with patch.object(archive.subprocess, 'run', side_effect=AssertionError('dry run contacted remote')):
                    self.assertFalse(archive.archive(live, 'test:', False)['applied'])
                with patch.object(archive.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'rclone')):
                    with self.assertRaises(subprocess.CalledProcessError):
                        archive.archive(live, 'test:', True)
                self.assertEqual(manifest.read_bytes(), before)
                self.assertFalse(scene.is_symlink())
                with patch.object(archive.subprocess, 'run', side_effect=[
                    subprocess.CompletedProcess('rclone', 0), subprocess.CalledProcessError(1, 'rclone check')
                ]):
                    with self.assertRaises(subprocess.CalledProcessError):
                        archive.archive(live, 'test:', True)
                self.assertEqual(manifest.read_bytes(), before)
                self.assertFalse(scene.is_symlink())
                with patch.object(archive.subprocess, 'run', side_effect=rclone):
                    first = archive.archive(live, 'test:', True)
                    self.assertEqual(first, archive.archive(live, 'test:', True))
                self.assertTrue(scene.is_symlink())
                self.assertEqual(scene.read_bytes(), b'generated scene')
                self.assertFalse(take.is_symlink())
                self.assertEqual(take.read_bytes(), b'immutable recording')
                self.assertEqual(len(json.loads(manifest.read_text())['files']), 2)


if __name__ == '__main__':
    unittest.main()

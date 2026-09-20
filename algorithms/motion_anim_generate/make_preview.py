"""Extract clean Landau views from existing comparison videos, retaining originals."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SANDBOX = Path(__file__).resolve().parent
CLOUD = Path(os.environ.get('GEO_CLOUD_ROOT', str(Path.home() / 'Nextcloud/Projects/geo_lib'))) / 'remote_outputs'


def main():
    manifest = json.loads((SANDBOX / 'gui/manifest.json').read_text())
    artifacts = manifest.get('artifacts', []) + [a for e in manifest.get('examples', []) for a in e.get('artifacts', [])]
    count = 0
    for relative in sorted({a['path'] for a in artifacts if a.get('kind') == 'video' and a['path'].endswith('/proof.mp4')}):
        candidates = [p for p in (ROOT / relative, CLOUD / relative) if p.is_file()]
        if not candidates:
            continue
        source = max(candidates, key=lambda p: p.stat().st_mtime)
        native = relative.replace('/proof.mp4', '/clean_preview.mp4')
        if any((root / native).is_file() for root in (ROOT, CLOUD)):
            # Native full-body render takes precedence over this legacy crop.
            continue
        target = ROOT / relative.replace('/proof.mp4', '/preview.mp4')
        stamp = target.with_suffix('.json')
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if target.exists() and stamp.exists() and json.loads(stamp.read_text()).get('source_sha256') == digest:
            continue
        info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'json', str(source)], text=True))['streams'][0]
        # The comparison renderer's lower-left panel is Landau's front view.
        if (info['width'], info['height']) != (960, 960):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name('preview.tmp.mp4')
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(source), '-vf', 'crop=468:348:6:542,scale=936:696', '-an', '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporary)], check=True)
        temporary.replace(target)
        stamp.write_text(json.dumps({'source': relative, 'source_sha256': digest, 'operation': 'Landau front view crop; original frames and timing retained'}, indent=2) + '\n')
        count += 1
    print(json.dumps({'clean_previews_updated': count}))


if __name__ == '__main__':
    main()

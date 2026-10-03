"""Check the actual exported running clip with the repository's Three.js."""
from pathlib import Path
import tempfile
import subprocess

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
with tempfile.TemporaryDirectory(prefix='landau-motion-check-') as temp:
    dst=Path(temp);vendor=REPO/'webgui/static/vendor/three'
    (dst/'three.mjs').write_bytes((vendor/'three.module.js').read_bytes())
    for src,name in [(vendor/'modules/GLTFLoader.js','loader'),(vendor/'modules/BufferGeometryUtils.js','utils'),
                     (ROOT/'gui/body-placement.js','placement'),(ROOT/'gui/motion-player.js','motion')]:
        text=src.read_text().replace("from 'three'","from './three.mjs'").replace("from 'BufferGeometryUtils'","from './utils.mjs'")
        (dst/(name+'.mjs')).write_text(text)
    (dst/'check.mjs').write_bytes((ROOT/'gui/motion-check.mjs').read_bytes())
    subprocess.run(['node',str(dst/'check.mjs'),str(ROOT/'outputs/landau_v10/landau_character.glb'),
                    str(ROOT/'outputs/landau_v10/motion_editor_validation.json')],check=True)

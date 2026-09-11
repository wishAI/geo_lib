"""Exercise body rest-space edits on the real GLB using the bundled Three.js."""
from pathlib import Path
import tempfile,subprocess,sys
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]
asset=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'outputs/landau_v10/landau_character.glb'
report=Path(sys.argv[2]) if len(sys.argv)>2 else ROOT/'outputs/landau_v10/editor_placement_validation.json'
with tempfile.TemporaryDirectory(prefix='landau-editor-check-') as temp:
    dst=Path(temp);vendor=REPO/'webgui/static/vendor/three'
    (dst/'three.mjs').write_bytes((vendor/'three.module.js').read_bytes())
    for src,name in [(vendor/'modules/GLTFLoader.js','loader'),(vendor/'modules/BufferGeometryUtils.js','utils'),(ROOT/'gui/body-placement.js','placement')]:
        text=src.read_text().replace("from 'three'","from './three.mjs'").replace("from 'BufferGeometryUtils'","from './utils.mjs'")
        (dst/(name+'.mjs')).write_text(text)
    (dst/'check.mjs').write_bytes((ROOT/'gui/placement-check.mjs').read_bytes())
    subprocess.run(['node',str(dst/'check.mjs'),str(asset),str(report)],check=True)

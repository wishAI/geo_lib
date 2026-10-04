"""Exercise connected fitting against the current GLB, including running poses."""
from pathlib import Path
import subprocess
import tempfile
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
with tempfile.TemporaryDirectory(prefix='landau-garment-fit-') as temp:
    dst=Path(temp);vendor=REPO/'webgui/static/vendor/three'
    (dst/'three.mjs').write_bytes((vendor/'three.module.js').read_bytes())
    for src,name in [(vendor/'modules/GLTFLoader.js','loader'),(vendor/'modules/BufferGeometryUtils.js','utils'),
                     (ROOT/'gui/body-placement.js','placement'),(ROOT/'gui/editing-pose.js','editing'),(ROOT/'gui/body-transition.js','transition'),(ROOT/'gui/motion-player.js','motion'),(ROOT/'gui/garment-fit.js','fit'),(ROOT/'gui/uv-regions.js','uv-regions'),(ROOT/'gui/vendor/GLTFExporter.js','exporter'),(ROOT/'gui/vendor/TextureUtils.js','texture')]:
        text=src.read_text().replace("from 'three'","from './three.mjs'").replace("from 'BufferGeometryUtils'","from './utils.mjs'")
        text=text.replace('/api/artifact?path=algorithms/3d_char_details/gui/vendor/TextureUtils.js','./texture.mjs')
        text=text.replace('/api/artifact?path=algorithms/3d_char_details/gui/uv-regions.js','./uv-regions.mjs')
        (dst/(name+'.mjs')).write_text(text)
    (dst/'check.mjs').write_bytes((ROOT/'gui/garment-fit-check.mjs').read_bytes())
    subprocess.run(['node',str(dst/'check.mjs'),str(ROOT/'outputs/landau_v10/landau_character.glb'),
        str(ROOT/'outputs/landau_v10/asset_report.json'),str(ROOT/'outputs/landau_v10/garment_fit_validation.json')],check=True)

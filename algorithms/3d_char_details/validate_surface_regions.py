"""Check shared face/body polygon fields on the current character GLB."""
from pathlib import Path
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
with tempfile.TemporaryDirectory(prefix='landau-surface-regions-') as temp:
    dst=Path(temp);vendor=REPO/'webgui/static/vendor/three'
    (dst/'three.mjs').write_bytes((vendor/'three.module.js').read_bytes())
    sources=[(vendor/'modules/GLTFLoader.js','loader'),(vendor/'modules/BufferGeometryUtils.js','utils')]
    sources += [(ROOT/('gui/'+name+'.js'),name) for name in ['surface-regions','influence-view','uv-regions','body-placement','body-transition','facial-controls']]
    sources += [(ROOT/'gui/vendor/GLTFExporter.js','exporter'),(ROOT/'gui/vendor/TextureUtils.js','texture')]
    for source,name in sources:
        text=source.read_text().replace("from 'three'","from './three.mjs'").replace("from 'BufferGeometryUtils'","from './utils.mjs'")
        text=text.replace('/api/artifact?path=algorithms/3d_char_details/gui/uv-regions.js','./uv-regions.mjs').replace('/api/artifact?path=algorithms/3d_char_details/gui/vendor/TextureUtils.js','./texture.mjs')
        (dst/(name+'.mjs')).write_text(text)
    (dst/'check.mjs').write_bytes((ROOT/'gui/surface-regions-check.mjs').read_bytes())
    subprocess.run(['node',str(dst/'check.mjs'),str(ROOT/'outputs/landau_v10/landau_character.glb'),str(ROOT/'outputs/landau_v10/asset_report.json'),str(ROOT/'outputs/landau_v10/editor_uv_atlas.json')],check=True)

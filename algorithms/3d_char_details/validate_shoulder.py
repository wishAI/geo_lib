"""Measure shoulder deformation using an explicitly supplied editor preset."""
from pathlib import Path
import argparse
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="landau-shoulder-check-") as directory:
        dst = Path(directory)
        vendor = REPO / "webgui/static/vendor/three"
        (dst / "three.mjs").write_bytes((vendor / "three.module.js").read_bytes())
        sources = [(vendor / "modules/GLTFLoader.js", "loader"),
                   (vendor / "modules/BufferGeometryUtils.js", "utils")]
        sources += [(ROOT / ("gui/" + source + ".js"), target) for source, target in
                    [("body-placement", "placement"), ("editing-pose", "editing"),
                     ("body-transition", "transition"), ("motion-player", "motion"),
                     ("garment-fit", "fit")]]
        for source, target in sources:
            text = source.read_text().replace("from 'three'", "from './three.mjs'")
            text = text.replace("from 'BufferGeometryUtils'", "from './utils.mjs'")
            # Local deformation helpers can be shared by the editor and checker.
            (dst / (target + ".mjs")).write_text(text)
        (dst / "check.mjs").write_bytes((ROOT / "gui/shoulder-check.mjs").read_bytes())
        command = ["node", str(dst / "check.mjs"),
                   str(ROOT / "outputs/landau_v10/landau_character.glb"),
                   str(ROOT / "outputs/landau_v10/asset_report.json"), str(args.preset)]
        if args.output:
            command.append(str(args.output))
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

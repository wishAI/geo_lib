# Minecraft Building Mapper — 2D Top Map

This isolated sandbox reads Minecraft Java Anvil worlds and renders the highest
non-air block visible at each X/Z coordinate. It is intentionally read-only: it
does not launch Minecraft, load Forge/Fabric, write NBT, invoke a data fixer, or
replace a region file.

Current scope is only the base 2D map:

- chunk-streamed 256×256-block PNG tiles
- exact top-face textures when the supplied asset evidence resolves them
- pan/zoom plus world, dimension, layer-cutoff, and texture-scale selectors
- machine-readable `metadata.json` and `report.json`
- legacy 1.5.2 `Blocks` + `Data` + optional `Add` decoding
- modern namespaced paletted section decoding, including negative section Y

Minecraft 26.1 moved every default dimension under
`dimensions/minecraft/<dimension>`; discovery supports that layout as well as
older and custom layouts. See the official [26.1 world-storage
notes](https://www.minecraft.net/en-us/article/minecraft-java-edition-26-1)
and [Fabric 26.1 compatibility
notes](https://www.fabricmc.net/2026/03/14/261.html).

There is no region-selection or building-recognition feature or control. Output
metadata reserves only a versioned, empty `extensions` object for future
file/API consumers.

## Data safety and placement

Do not copy or hydrate `LunaMatrix_20260429.zip`, either world save, `bc3`, mod
JARs, or texture packs onto the Mac. Keep them on UGREEN, Nextcloud, or TK2.
The known UGREEN bundle contains `Bilicraft三周目地图.zip`, `bc3.zip`, and
`bc3 for mac.zip`.

Inspect the outer archive in place before extracting anything:

```bash
python3 -m algorithms.minecraft_building_mapper inspect-archive \
  --archive /remote/storage/LunaMatrix_20260429.zip \
  --output /remote/storage/minecraft-map/archive_inventory.json
```

Nested ZIP members cannot support random region seeks. On UGREEN or TK2 only,
extract the inner world ZIP and the matching `bc3` instance into a dedicated
working directory. Preserve the original archives and the extracted Bilicraft
world unchanged. Point the mapper either at the extracted world directory or at
the inner world ZIP. Never open Bilicraft with another Minecraft version.

## Inspect worlds

```bash
python3 -m algorithms.minecraft_building_mapper inspect-world \
  --world /remote/storage/worlds/Bilicraft三周目地图.zip
```

The command reports the save's `DataVersion`, name, and discovered dimensions.
Normal dimensions are `minecraft:overworld`, `minecraft:the_nether`, and
`minecraft:the_end`; modern `dimensions/<namespace>/<path>` folders are also
discovered.

## Build exact texture catalogs

Asset order is highest precedence first, like Minecraft resource packs. Catalogs
contain paths and SHA-256 hashes, not copied textures.

For LunaMatrix Fabric 26.1.2, pass its exact enabled resource packs, mod JARs,
and game JAR. The resolver follows blockstates, model parents, and top-face
texture references. Multipart geometry or ambiguous weighted top textures stay
unknown rather than being guessed.

```bash
python3 -m algorithms.minecraft_building_mapper catalog-modern \
  --asset /remote/storage/LunaMatrix/resourcepacks/enabled-pack.zip \
  --asset /remote/storage/LunaMatrix/mods/exact-mod.jar \
  --asset /remote/storage/versions/26.1.2/26.1.2.jar \
  --output /remote/storage/minecraft-map/lunamatrix_catalog.json
```

For Bilicraft, use the exact runtime `bc3` tree (not a newly generated Forge
instance), the BiliCraft 1.5.1 32× texture pack, the
`CustomStuff/BilicraftMOD` texture directory, and every exact mod JAR. The
catalog reads numeric assignments from the Forge configs and ID/metadata/texture
fields from CustomStuff definitions. Vanilla 1.5.2 IDs use canonical atlas
coordinates but pixels always come from the supplied 32× pack. A config-only ID
is named with its exact provenance but remains texture-unknown because config
assignment alone does not prove a metadata-specific top face.

```bash
python3 -m algorithms.minecraft_building_mapper catalog-legacy \
  --bc3-root /remote/storage/LunaMatrix/bc3 \
  --asset '/remote/storage/LunaMatrix/BiliCraft 1.5.1 32x.zip' \
  --asset /remote/storage/LunaMatrix/bc3/mods/CustomStuff/BilicraftMOD \
  --asset /remote/storage/LunaMatrix/bc3/mods/exact-mod.jar \
  --output /remote/storage/minecraft-map/bilicraft_catalog.json
```

Every unresolved palette key is drawn as a deterministic black/magenta checker
and counted under `unknown.blockCounts` in the report. Do not hand-map an
unknown block from appearance or a similarly named mod.

## Validate one tile first

A tile is a fixed streaming unit, not a user region-selection feature. Tile
`0,0` covers X `[0,256)` and Z `[0,256)` and loads at most 16×16 chunks.

```bash
python3 -m algorithms.minecraft_building_mapper render-tile \
  --world /remote/storage/worlds/Bilicraft三周目地图.zip \
  --catalog /remote/storage/minecraft-map/bilicraft_catalog.json \
  --dimension minecraft:overworld --layer surface \
  --tile-x 0 --tile-z 0 --pixels-per-block 4 \
  --output algorithms/minecraft_building_mapper/outputs/bilicraft_smoke
```

The output folder contains `tile_0_0.png`, `metadata.json`, and `report.json`.
Keep it on TK2 or pull the ignored output through `./geo pull-output
minecraft_building_mapper`, which places remote results in the established
Nextcloud workflow.

## Pan and zoom viewer

Each `--world` label must have one matching `--catalog` label. The browser loads
only visible tiles and exposes no save mutation endpoint.

```bash
python3 -m algorithms.minecraft_building_mapper serve \
  --world LunaMatrix=/remote/storage/worlds/LunaMatrix \
  --catalog LunaMatrix=/remote/storage/minecraft-map/lunamatrix_catalog.json \
  --world Bilicraft=/remote/storage/worlds/Bilicraft三周目地图.zip \
  --catalog Bilicraft=/remote/storage/minecraft-map/bilicraft_catalog.json \
  --cache algorithms/minecraft_building_mapper/outputs/tile_cache \
  --host 127.0.0.1 --port 8782
```

Open `http://127.0.0.1:8782`. Enter `surface` or an integer Y cutoff in the
layer field. Drag to pan and use the wheel/trackpad to zoom.

## Tests

Tests synthesize tiny NBT/Anvil fixtures under the system temporary directory;
they do not contain or touch either real save.

```bash
python3 -m pytest algorithms/minecraft_building_mapper/tests -q
./geo storage audit
```

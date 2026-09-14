# Minecraft Building Mapper — 2D Top Map

This isolated sandbox reads Minecraft Java Anvil worlds and renders the highest
non-air block visible at each X/Z coordinate. It is intentionally read-only: it
does not launch Minecraft, load Forge/Fabric, write NBT, invoke a data fixer, or
replace a region file.

Current scope is only the base 2D map:

- chunk-streamed 256×256-block PNG tiles
- complete allocated-chunk inventory and bounded whole-world overview rendering
- exact top-face textures when the supplied asset evidence resolves them
- alpha-aware substrate composition, representative fluid depth, partial-block geometry, and bounded height shading
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

Dimension discovery supports arbitrary path depth below
`dimensions/<namespace>/`, not only the single-segment default dimension
names. The validated LunaMatrix full pack contains such a separate
`minecraft:custom/resource` dimension, but it is intentionally excluded from
the LunaMatrix overworld deliverable.

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
texture references. Multipart models that share a top texture are exact;
multi-sprite multipart models use one exact supplied representative only with
an explicit `inferred` label. Mojang client artifacts must be selected through
the matching authoritative version manifest and hash-verified.

```bash
python3 -m algorithms.minecraft_building_mapper catalog-modern \
  --asset /remote/storage/LunaMatrix/resourcepacks/enabled-pack.zip \
  --asset /remote/storage/LunaMatrix/mods/exact-mod.jar \
  --asset /remote/storage/versions/26.1.2/26.1.2.jar \
  --output /remote/storage/minecraft-map/lunamatrix_catalog.json
```

The verified LunaMatrix fullpack world lives at `lunamatrix-test/world` and is
DataVersion 4790 / Minecraft 26.1.2. Its server JAR does not contain the full
client texture set. For the validated run, Mojang's official manifest selected
client SHA-1 `4e618f09a0c649dde3fdf829df443ce0b8831e65` (SHA-256
`b1b3158572666445eff01e82fad8c7de2e4953db6d354f311730d77a8359d0b0`).
The exact fullpack server and mod JARs remain additional catalog sources.

For Bilicraft, use the exact runtime `bc3` tree (not a newly generated Forge
instance), the BiliCraft 1.5.1 32× texture pack, the
`CustomStuff/BilicraftMOD` texture directory, and every exact mod JAR. The
catalog reads numeric assignments from the Forge configs and ID/metadata/texture
fields from CustomStuff definitions. Fingerprint-gated legacy resolvers then
combine exact config IDs, bundled registration/render bytecode, client options,
language labels, and texture members. The resolver refuses its hard-coded
bytecode summary unless the complete mod archive SHA-256 matches the audited
bc3 copy. Vanilla 1.5.2 IDs use canonical mappings but pixels always come from
the supplied 32× pack. The exact client biome table and ForgottenNature's
fingerprinted biome classes/config provide temperature and rainfall for IDs
0–22 and 70–76. A config-only ID is named with its exact provenance but
remains texture-unknown because config assignment alone does not prove a
metadata-specific top face.

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

After a complete read-only legacy inventory, explicitly cover only its visible
unresolved states (and unresolved metadata in the same exactly configured ID
families) with audited representative textures:

```bash
python3 -m algorithms.minecraft_building_mapper catalog-cover-visible \
  --catalog /remote/storage/minecraft-map/bilicraft_exact_catalog.json \
  --inventory /remote/storage/minecraft-map/bilicraft_inventory/surface_inventory.json \
  --output /remote/storage/minecraft-map/bilicraft_render_catalog.json
```

This command never upgrades confidence to `exact`: every fallback records the
triggering surface count, exact config evidence when present, representative
texture member/hash, reason, and confidence below 1.0.

### Legacy satellite rendering

The renderer follows the exact bundled OptiFine settings (`customColors`,
`smoothBiomes`, and `swampColors`) and its 3×3 smoothing rule. Grass, foliage,
pine, birch, swamp, water, lily, stem, and configured block palettes use the
supplied pack's exact colorizer pixels. Animated texture strips use their first
square frame. Transparent surfaces are composited over the next visible layer;
water retains the visible substrate while darkening with measured column depth.

Crossed plants/crops, lines and rails, connected fences/panes, vertical planes,
slabs, stairs, snow covers, partial blocks, leaves, glass, ice, and fluids have
explicit overhead geometry categories. The report counts every contributing
state by geometry and certainty. A bounded directional height shade is applied
after rendering to make terrain and roof edges legible; it changes pixels only,
never block identity or world data.

### Resolution and inference contract

Every catalog entry carries `resolution`, numeric `confidence`, `reason`, and a
machine-readable `provenance` list:

- `exact`: the supplied runtime config/code/options and supplied pixels directly
  prove the result; confidence is `1.0`.
- `resolved`: a stable external format mapping, such as a canonical vanilla
  numeric ID, selects pixels from an exact supplied asset; confidence is `1.0`.
- `inferred`: exact evidence is unavailable and a private/internal mapping is
  explicitly inferred; confidence must be greater than zero and less than one.
- `unknown`: no appearance is asserted; confidence is `0.0`.

The renderer reports visible counts for all four categories. An exact class may
also prove that a block is non-rendering: Railcraft `block.hidden` is recorded as
`renderAsAir`, skipped while scanning downward, and counted separately under
`skippedAsAir`. No inference is used for that block.

### Audited Bilicraft evidence

The exact pack inspected on TK2 has these stable hashes:

| Input | SHA-256 |
|---|---|
| `bc3.zip` | `dc7274588b2a4fc110c9f8dd9301c68a69242da3d7cfe2d6ad45c7ab7751967c` |
| `ForgottenNature for 1.5.2.zip` (embedded version string `1.2.9`) | `063cc073a6fb18990073ea632c151526ce8fb24bb14bdcc13c15702e46176ee3` |
| `ForgottenNature.cfg` | `2cdd2787f679aa245a6b6b3fe990627702e3ad3b55ee4fefb75896ff3c753531` |
| `.minecraft/options.txt` (`fancyGraphics:true`) | `3fddfb91d6c4a3c1ceec1ea88fbc173642ce6c0a0710c55f1241bcee2b2d2f49` |
| `BiliCraft_1.5.1_32x.zip` | `2dc23a6b2c3eced8a2645054586c6b80f82525cdda9b0cf9090679e56acf4296` |
| `Railcraft_1.5.2-7.2.3.0.jar` | `42b6b736a544303eafc420504e6097c1a60ac1e3e9eeb16d9217a05b209438c2` |

The first exact mappings are `leafIDindex=4084` plus contiguous leaf classes,
`logIDindex=4079` plus contiguous log classes, and `FlowerID=4092`. Stored leaf
metadata `8..15` has its decay bit removed by the exact class before selecting
the icon. The exact `fancyGraphics:true` client setting selects non-`Solid`
leaf textures. In the validation tile this proves, among others, Fig/Cypress at
4084, Acacia at 4085, Poplar at 4086, the metadata-indexed ten-flower table at
4092, and the log cross-section for 4079:8.

### Reproducible TK2 acquisition and extraction

Run these only on TK2. The save remains remote; do not run this workflow in a
Mac checkout and do not open the save with Minecraft.

```bash
cache=$(mktemp -d /tmp/minecraft-legacy-resolver.XXXXXX)
rclone copyto \
  'ugreen-nextcloud:Documents/Games/Minecraft/Bilicraft-1.5.2/from-ugreen/bc3.zip' \
  "$cache/bc3.zip"
sha256sum "$cache/bc3.zip"
unzip -tq "$cache/bc3.zip"

# Select only evidence needed by the catalog into TK2 cache.
unzip -q "$cache/bc3.zip" -d "$cache/exact" \
  'bc3/.minecraft/config/*' 'bc3/.minecraft/mods/*' \
  'bc3/.minecraft/coremods/*' 'bc3/.minecraft/texturepacks/*' \
  'bc3/.minecraft/options.txt'
```

Normalize the extracted `bc3/.minecraft` directory as `--bc3-root`; pass the
texture pack first, then `CustomStuff/mods/BilicraftMOD`, then the exact mod and
coremod archives. The catalog hashes every asset and refuses to reopen it if an
asset changes.

For historical comparison, the original author’s [Minecraft Forum release
record](https://www.minecraftforum.net/forums/mapping-and-modding-java-edition/minecraft-mods/1286390-forgotten-nature-1-7-19-a-natural-addition-to-mc)
identifies ForgottenNature 1.2.9 for Minecraft 1.5.1 and documents six
contiguous leaf IDs; the exact bundled binary’s embedded version is also 1.2.9.
The forum’s original Dropbox binary link now returns HTTP 403, so no external
binary hash is claimed. The author-associated [ForgottenNature v1.3.0 source
repository](https://github.com/alexandrage/ForgottenNature_v1.3.0) at commit
`65069082fe537158fd0d64092399fc453f658b39` was downloaded to TK2 cache as a
source archive (`cdef3358b78509057d575ae45eced4efd948813fc54c3ff58c89229dfd65062f`).
It targets later Minecraft 1.6.2–1.6.4, so it is comparison evidence only. Its
ID-offset and icon tables agree with the exact 1.2.9 bytecode, which remains the
authority used by the resolver.

Storage lineage note: the standalone `bc3.zip` was intentionally moved by the
parent orchestration from UGREEN Backup to the Nextcloud path above. Nextcloud
chunk upload was assembled first, source and destination SHA-256 both matched
`dc7274588b2a4fc110c9f8dd9301c68a69242da3d7cfe2d6ad45c7ab7751967c`,
and only then was the standalone source deleted. Its disappearance is expected,
not an SMB anomaly. `LunaMatrix_20260429.zip` still contains the second copy.

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

## Inventory and render the whole explored world

These commands discover allocated chunks themselves. They do not accept a
region and do not expose the QA density ranking as building recognition.

```bash
python3 -m algorithms.minecraft_building_mapper inventory-world \
  --world /remote/storage/worlds/Bilicraft三周目地图.zip \
  --catalog /remote/storage/minecraft-map/bilicraft_catalog.json \
  --dimension minecraft:overworld \
  --output /remote/storage/minecraft-map/bilicraft_inventory

python3 -m algorithms.minecraft_building_mapper render-overview \
  --world /remote/storage/worlds/Bilicraft三周目地图.zip \
  --catalog /remote/storage/minecraft-map/bilicraft_catalog.json \
  --dimension minecraft:overworld --max-size 4096 \
  --output /remote/storage/minecraft-map/bilicraft_overview
```

`surface_inventory.json` includes explored bounds, biome and top-state counts,
all transparent surface contributions, geometry totals, certainty coverage,
non-rendering helpers, and parse errors. `world_overview.png` and
`overview_report.json` record the automatically chosen blocks-per-pixel scale,
source bounds, tint diagnostics, texture failures, and shade range. The QA-only
constructed-material density list may be used by an operator to choose an
existing fixed 256×256 tile for closer visual inspection.

The overview canvas uses the dominant four-neighbour explored chunk component.
Disconnected coordinate-jump components remain counted with full bounds under
`disconnectedComponentsQuarantinedFromCanvas`; this keeps the inhabited world
navigable without deleting, rewriting, or hiding evidence from the inventory.

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

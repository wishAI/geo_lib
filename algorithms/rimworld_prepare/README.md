# RimWorld Prepare

Local WebGUI and standalone builder for a clean Dragon Yuran race mod. The
source inventory is read from the Yuran Race workshop installation on TK2;
generated source caches, previews, mod packages, and character presets stay in
`outputs/` and are intentionally not committed.

## Workflow

1. `python3 algorithms/rimworld_prepare/builder.py sync`
2. Open `./geo gui` and choose **RimWorld Prepare**.
3. Select apparel, configure the Dragon Bone giant multiplier, edit colonists,
   and optionally export/import the dragon-skin
   img2img sheet. Its three tightly framed 640px cells join the body and
   hairless face into one continuous figure for South, East, and North;
   RimWorld mirrors East for West.
4. `python3 algorithms/rimworld_prepare/builder.py deploy`

Deployment writes the mod to TK2's RimWorld `Mods/DragonYuran` directory and
Prepare Carefully v5 files to its SaveData `PrepareCarefully` directory. It
does not launch RimWorld or edit the active mod list.

## Dragon Bone dependency contract

The cybernetic Dragon Bone is installed on a Dragon Yuran's spine. It grants a
manual giant-form command; cancelling the resulting status transforms the pawn
back. The exact body-size multiplier is configurable from 1.25x to 10.00x in
the WebGUI and defaults to 5.00x.

Dragon Yuran depends on Harmony, Humanoid Alien Races, and **Big and Small -
Framework** (`RedMattis.BetterPrerequisites`). The implementation uses the
Framework's ability, cancellable-Hediff, and `SM_BodySizeMultiplier` interfaces.
It neither depends on nor references **Big and Small - Genes & More**
(`RedMattis.BigSmall.Core`).

## Rendering contract

The preview uses the RimWorld 1.6 `Thin.headOffset`, Yuran 1.6
`headOffsetDirectional`, and HAR named plus directional body-addon offsets.
West textures are mirrored from east only when no authored west texture exists.
The WebGUI is a deterministic texture-layer preview; final Unity rendering must
be checked the next time the user elects to launch the game.

An imported img2img sheet is preserved as a checksum-addressed run. The builder
restores all three joined views to their original 256x256 composite coordinates,
projects them through the six original body/head alpha silhouettes, and emits per-asset original/generated/difference
images and observed pixel-change metrics in the Dragon Skin tab. The active
Dragon Skin body and face are promoted only by an explicit upload.

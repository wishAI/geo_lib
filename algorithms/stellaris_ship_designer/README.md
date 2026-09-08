# Stellaris Ship Designer

This local-only sandbox previews editable browser conversions of assets from the user's installed copy of Stellaris on TK2. It does not contain generated stand-in ship geometry.

## Included source examples

- **Mammalian Battleship** — the original XL1 bow, L3 core, L1 stern, five-bone frame, 39 locators, and the shipped `idle`, `death`, `death2`, and `death3` actions.
- **BioGenesis Mauler, growth stage 1** — the original skinned mesh, 16-bone hierarchy, 10 embedded locators, one source-defined `root` slot-origin marker, and the shipped `idle`, `combat_moving`, and `attack` action files. The installed `biogenesis_01_ships.gfx` binds idle and combat movement. Its attack binding is commented out, so the designer labels that clip **shipped but disabled**.

The Mauler mesh contains `weapon_01` and `weapon_02`, but the installed `MAULER_STAGE_1_swarm` section binds both weapon components to `locatorname = "root"` through invisible turret templates. The designer therefore shows `root` as the official slot origin and labels the two mesh weapon locators as embedded but unbound. Locator XYZ lines are local transform axes, not projectile paths; the turret and projectile definitions determine firing direction.

The optional **Laser test** is a sandbox visualization, not an extracted game action. Battleship muzzle markers animate a cyan shot along their pose-derived local `+Z` rest direction. The Mauler stage-1 `root` binding has no fixed direction in the section source, so its test displays an expanding origin pulse until a firing axis is explicitly chosen. Distant target, engine, exhaust, and XL wind-up markers remain preserved but are hidden by default; selecting any locator temporarily reveals its exact source pose.

The BioGenesis size definition identifies the Mauler stage 1 as a `bio_ship`, gives it `fleet_slot_size = 1`, and upgrades it to stage 2. The swarm section template has torpedo and small-gun component slots, two small utility slots, and one auxiliary slot.

## Provenance and storage

The PDX `.mesh` and `.anim` files were decoded with `io_pdx_mesh` commit `170a4cba0c272c3825e79bbba244796fdc746e92`, then exported as GLB with their original skinning, hierarchy, locator nodes, texture maps, and actions intact. Collision-only geometry was removed for the browser preview. Battleship section locators were parented to their corresponding original frame bones so they follow the shipped death actions.

The GLBs are intentionally excluded from Git. `large_files.json` declares their exact byte sizes and SHA-256 hashes under the project Nextcloud asset store, while the repository input paths are stable symlinks created by `./geo storage hydrate`.

Only use the extracted assets with a legitimately installed copy of Stellaris and subject to the game's applicable terms.

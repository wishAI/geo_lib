"""Minecraft Java 1.5.2 top-face and biome render rules.

The tables mirror the exact bundled client's block registrations and biome
climates. Pixel data still comes from the supplied resource pack or mod JARs.
"""

from __future__ import annotations

from typing import Any


EXACT_BC3_CLIENT_SHA256 = "abfda8f7aab46aa5e6665b3991ac670959d7e05162025b9ec89e6077577efe7b"
FORGOTTEN_NATURE_152_SHA256 = "063cc073a6fb18990073ea632c151526ce8fb24bb14bdcc13c15702e46176ee3"

# ID, display name, temperature, rainfall, water multiplier. Values are from
# aav.class in the fingerprinted bc3 client. Unspecified water is white.
VANILLA_152_BIOMES: dict[int, tuple[str, float, float, int]] = {
    0: ("Ocean", 0.5, 0.5, 0xFFFFFF),
    1: ("Plains", 0.8, 0.4, 0xFFFFFF),
    2: ("Desert", 2.0, 0.0, 0xFFFFFF),
    3: ("Extreme Hills", 0.2, 0.3, 0xFFFFFF),
    4: ("Forest", 0.7, 0.8, 0xFFFFFF),
    5: ("Taiga", 0.05, 0.8, 0xFFFFFF),
    6: ("Swampland", 0.8, 0.9, 0xE0FFAE),
    7: ("River", 0.5, 0.5, 0xFFFFFF),
    8: ("Hell", 2.0, 0.0, 0xFFFFFF),
    9: ("Sky", 0.5, 0.5, 0xFFFFFF),
    10: ("FrozenOcean", 0.0, 0.5, 0xFFFFFF),
    11: ("FrozenRiver", 0.0, 0.5, 0xFFFFFF),
    12: ("Ice Plains", 0.0, 0.5, 0xFFFFFF),
    13: ("Ice Mountains", 0.0, 0.5, 0xFFFFFF),
    14: ("MushroomIsland", 0.9, 1.0, 0xFFFFFF),
    15: ("MushroomIslandShore", 0.9, 1.0, 0xFFFFFF),
    16: ("Beach", 0.8, 0.4, 0xFFFFFF),
    17: ("DesertHills", 2.0, 0.0, 0xFFFFFF),
    18: ("ForestHills", 0.7, 0.8, 0xFFFFFF),
    19: ("TaigaHills", 0.05, 0.8, 0xFFFFFF),
    20: ("Extreme Hills Edge", 0.2, 0.3, 0xFFFFFF),
    21: ("Jungle", 1.2, 0.9, 0xFFFFFF),
    22: ("JungleHills", 1.2, 0.9, 0xFFFFFF),
}


VANILLA_NAMES = {
    1: "stone", 2: "grass", 3: "dirt", 4: "cobblestone", 5: "planks", 6: "sapling", 7: "bedrock",
    8: "flowing_water", 9: "water", 10: "flowing_lava", 11: "lava", 12: "sand", 13: "gravel",
    14: "gold_ore", 15: "iron_ore", 16: "coal_ore", 17: "log", 18: "leaves", 19: "sponge", 20: "glass",
    21: "lapis_ore", 22: "lapis_block", 23: "dispenser", 24: "sandstone", 25: "note_block", 26: "bed",
    27: "powered_rail", 28: "detector_rail", 29: "sticky_piston", 30: "cobweb", 31: "tall_grass",
    32: "dead_bush", 33: "piston", 34: "piston_head", 35: "wool", 36: "moving_piston", 37: "dandelion",
    38: "rose", 39: "brown_mushroom", 40: "red_mushroom", 41: "gold_block", 42: "iron_block",
    43: "double_stone_slab", 44: "stone_slab", 45: "bricks", 46: "tnt", 47: "bookshelf",
    48: "mossy_cobblestone", 49: "obsidian", 50: "torch", 51: "fire", 52: "mob_spawner", 53: "oak_stairs",
    54: "chest", 55: "redstone_wire", 56: "diamond_ore", 57: "diamond_block", 58: "crafting_table",
    59: "wheat", 60: "farmland", 61: "furnace", 62: "lit_furnace", 63: "standing_sign", 64: "wooden_door",
    65: "ladder", 66: "rail", 67: "cobblestone_stairs", 68: "wall_sign", 69: "lever", 70: "stone_pressure_plate",
    71: "iron_door", 72: "wooden_pressure_plate", 73: "redstone_ore", 74: "lit_redstone_ore",
    75: "unlit_redstone_torch", 76: "redstone_torch", 77: "stone_button", 78: "snow_layer", 79: "ice",
    80: "snow_block", 81: "cactus", 82: "clay", 83: "sugar_cane", 84: "jukebox", 85: "oak_fence",
    86: "pumpkin", 87: "netherrack", 88: "soul_sand", 89: "glowstone", 90: "nether_portal",
    91: "jack_o_lantern", 92: "cake", 93: "unpowered_repeater", 94: "powered_repeater", 95: "locked_chest",
    96: "trapdoor", 97: "monster_egg", 98: "stone_bricks", 99: "brown_mushroom_block", 100: "red_mushroom_block",
    101: "iron_bars", 102: "glass_pane", 103: "melon", 104: "pumpkin_stem", 105: "melon_stem", 106: "vine",
    107: "fence_gate", 108: "brick_stairs", 109: "stone_brick_stairs", 110: "mycelium", 111: "lily_pad",
    112: "nether_bricks", 113: "nether_brick_fence", 114: "nether_brick_stairs", 115: "nether_wart",
    116: "enchanting_table", 117: "brewing_stand", 118: "cauldron", 119: "end_portal", 120: "end_portal_frame",
    121: "end_stone", 122: "dragon_egg", 123: "redstone_lamp", 124: "lit_redstone_lamp",
    125: "double_wood_slab", 126: "wood_slab", 127: "cocoa", 128: "sandstone_stairs", 129: "emerald_ore",
    130: "ender_chest", 131: "tripwire_hook", 132: "tripwire", 133: "emerald_block", 134: "spruce_stairs",
    135: "birch_stairs", 136: "jungle_stairs", 137: "command_block", 138: "beacon", 139: "cobblestone_wall",
    140: "flower_pot", 141: "carrots", 142: "potatoes", 143: "wooden_button", 144: "skull", 145: "anvil",
    146: "trapped_chest", 147: "light_weighted_pressure_plate", 148: "heavy_weighted_pressure_plate",
    149: "unpowered_comparator", 150: "powered_comparator", 151: "daylight_sensor", 152: "redstone_block",
    153: "nether_quartz_ore", 154: "hopper", 155: "quartz_block", 156: "quartz_stairs", 157: "activator_rail",
    158: "dropper",
}


def _wood(meta: int) -> str:
    return ("wood.png", "wood_spruce.png", "wood_birch.png", "wood_jungle.png")[meta & 3]


def _log(meta: int) -> str:
    if meta & 12 == 0:
        return "tree_top.png"
    return ("tree_side.png", "tree_spruce.png", "tree_birch.png", "tree_jungle.png")[meta & 3]


def _leaves(meta: int) -> str:
    return ("leaves.png", "leaves_spruce.png", "leaves.png", "leaves_jungle.png")[meta & 3]


def _slab(meta: int) -> str:
    return (
        "stoneslab_top.png", "sandstone_top.png", "wood.png", "stonebrick.png",
        "brick.png", "stonebricksmooth.png", "netherBrick.png", "quartzblock_top.png",
    )[meta & 7]


def vanilla_texture(block_id: int, metadata: int) -> dict[str, Any] | None:
    """Return an exact-pack member description for the upward view."""

    simple = {
        1: "stone.png", 2: "grass_top.png", 3: "dirt.png", 4: "stonebrick.png", 7: "bedrock.png",
        8: "water.png", 9: "water.png", 10: "lava.png", 11: "lava.png", 12: "sand.png", 13: "gravel.png",
        14: "oreGold.png", 15: "oreIron.png", 16: "oreCoal.png", 19: "sponge.png", 20: "glass.png",
        21: "oreLapis.png", 22: "blockLapis.png", 24: "sandstone_top.png", 25: "musicBlock.png",
        30: "web.png", 32: "deadbush.png", 34: "piston_inner_top.png", 37: "flower.png", 38: "rose.png",
        39: "mushroom_brown.png", 40: "mushroom_red.png", 41: "blockGold.png", 42: "blockIron.png",
        45: "brick.png", 46: "tnt_top.png", 47: "wood.png", 48: "stoneMoss.png", 49: "obsidian.png",
        50: "torch.png", 51: "fire_0.png", 52: "mobSpawner.png", 53: "wood.png", 55: "redstoneDust_cross.png",
        56: "oreDiamond.png", 57: "blockDiamond.png", 58: "workbench_top.png", 61: "furnace_top.png",
        62: "furnace_top.png", 63: "wood.png", 65: "ladder.png", 66: "rail.png", 67: "stonebrick.png",
        68: "wood.png",
        69: "lever.png", 70: "stone.png", 72: "wood.png", 73: "oreRedstone.png", 74: "oreRedstone.png",
        75: "redtorch.png", 76: "redtorch_lit.png", 77: "stone.png", 78: "snow.png", 79: "ice.png",
        80: "snow.png", 81: "cactus_top.png", 82: "clay.png", 83: "reeds.png", 84: "jukebox_top.png",
        85: "wood.png", 86: "pumpkin_top.png", 87: "hellrock.png", 88: "hellsand.png", 89: "lightgem.png",
        90: "portal.png", 91: "pumpkin_top.png", 92: "cake_top.png", 93: "repeater.png", 94: "repeater_lit.png",
        96: "trapdoor.png", 101: "fenceIron.png", 102: "thinglass_top.png", 103: "melon_top.png",
        106: "vine.png", 107: "wood.png", 108: "brick.png", 109: "stonebricksmooth.png", 110: "mycel_top.png",
        111: "waterlily.png", 112: "netherBrick.png", 113: "netherBrick.png", 114: "netherBrick.png",
        116: "enchantment_top.png", 117: "brewingStand.png", 118: "cauldron_top.png", 119: "../../misc/tunnel.png",
        120: "endframe_top.png",
        121: "whiteStone.png", 122: "dragonEgg.png", 123: "redstoneLight.png", 124: "redstoneLight_lit.png",
        128: "sandstone_top.png", 129: "oreEmerald.png", 131: "tripWireSource.png", 132: "tripWire.png",
        133: "blockEmerald.png", 134: "wood_spruce.png", 135: "wood_birch.png", 136: "wood_jungle.png",
        137: "commandBlock.png", 138: "beacon.png", 139: "stonebrick.png", 140: "flowerPot.png",
        143: "wood.png", 144: "stone.png", 147: "blockGold.png", 148: "blockIron.png", 149: "comparator.png",
        150: "comparator_lit.png", 151: "daylightDetector_top.png", 152: "blockRedstone.png", 153: "netherquartz.png",
        154: "hopper_top.png", 156: "quartzblock_top.png", 157: "activatorRail.png", 158: "furnace_top.png",
    }
    member = simple.get(block_id)
    if block_id == 5:
        member = _wood(metadata)
    elif block_id == 6:
        member = ("sapling.png", "sapling_spruce.png", "sapling_birch.png", "sapling_jungle.png")[metadata & 3]
    elif block_id == 17:
        member = _log(metadata)
    elif block_id == 18:
        member = _leaves(metadata)
    elif block_id == 23:
        member = "dispenser_front_vertical.png" if (metadata & 7) == 1 else "furnace_top.png"
    elif block_id == 26:
        member = "bed_head_top.png" if metadata & 8 else "bed_feet_top.png"
    elif block_id in (27, 28, 157):
        base = {27: "goldenRail", 28: "detectorRail", 157: "activatorRail"}[block_id]
        member = f"{base}_powered.png" if metadata & 8 else f"{base}.png"
    elif block_id in (29, 33):
        facing = metadata & 7
        member = ("piston_bottom.png" if facing == 0 else
                  ("piston_top_sticky.png" if block_id == 29 else "piston_top.png") if facing == 1 else "piston_side.png")
    elif block_id == 31:
        member = ("deadbush.png", "tallgrass.png", "fern.png")[metadata if metadata < 3 else 0]
    elif block_id == 35:
        member = f"cloth_{metadata & 15}.png"
    elif block_id in (43, 44):
        member = _slab(metadata)
    elif block_id == 54:
        return {"member": "item/chest.png", "crop": [14 / 64, 0.0, 14 / 64, 14 / 64]}
    elif block_id == 59:
        member = f"crops_{metadata & 7}.png"
    elif block_id == 60:
        member = "farmland_wet.png" if metadata & 7 == 7 else "farmland_dry.png"
    elif block_id == 64:
        member = "doorWood_upper.png" if metadata & 8 else "doorWood_lower.png"
    elif block_id == 71:
        member = "doorIron_upper.png" if metadata & 8 else "doorIron_lower.png"
    elif block_id == 97:
        member = ("stone.png", "stonebrick.png", "stonebricksmooth.png")[min(metadata & 3, 2)]
    elif block_id == 98:
        member = ("stonebricksmooth.png", "stonebricksmooth_mossy.png", "stonebricksmooth_cracked.png", "stonebricksmooth_carved.png")[metadata & 3]
    elif block_id == 99:
        member = "mushroom_skin_stem.png" if metadata == 10 else "mushroom_skin_brown.png"
    elif block_id == 100:
        member = "mushroom_skin_stem.png" if metadata == 10 else "mushroom_skin_red.png"
    elif block_id in (104, 105):
        member = "stem_bent.png" if metadata == 7 else "stem_straight.png"
    elif block_id == 115:
        member = f"netherStalk_{0 if metadata <= 2 else 1 if metadata == 3 else 2}.png"
    elif block_id in (125, 126):
        member = _wood(metadata)
    elif block_id == 127:
        member = f"cocoa_{min((metadata >> 2), 2)}.png"
    elif block_id == 130:
        return {"member": "item/enderchest.png", "crop": [14 / 64, 0.0, 14 / 64, 14 / 64]}
    elif block_id in (141, 142):
        stage = min(metadata >> 1, 3)
        member = f"{'carrots' if block_id == 141 else 'potatoes'}_{stage}.png"
    elif block_id == 145:
        member = f"anvil_top{'' if metadata >> 2 == 0 else '_damaged_' + str(min(metadata >> 2, 2))}.png"
    elif block_id == 146:
        return {"member": "item/chests/trap_small.png", "crop": [14 / 64, 0.0, 14 / 64, 14 / 64]}
    elif block_id == 155:
        member = ("quartzblock_top.png", "quartzblock_chiseled_top.png", "quartzblock_lines_top.png",
                  "quartzblock_lines.png", "quartzblock_lines.png")[min(metadata, 4)]
    if member is None:
        return None
    if member.startswith("../../"):
        return {"member": member.removeprefix("../../")}
    return {"member": f"textures/blocks/{member}"}


def vanilla_visual(block_id: int, metadata: int) -> dict[str, Any]:
    """Describe a top-down geometry/tint without changing the stored state."""

    geometry = "cube"
    height = 1.0
    tint = None
    if block_id in {6, 30, 31, 32, 37, 38, 39, 40, 51, 59, 75, 76, 83, 104, 105, 115, 127, 141, 142}:
        geometry = "cross"
    elif block_id in {50, 69, 77, 117, 131, 140, 144}:
        geometry = "point"
    elif block_id in {27, 28, 55, 66, 93, 94, 132, 149, 150, 157}:
        geometry = "line"
    elif block_id in {63, 64, 65, 68, 71, 90, 96, 106, 143}:
        geometry = "plane"
    elif block_id in {85, 101, 102, 107, 113, 139}:
        geometry = "connected"
    elif block_id in {8, 9, 10, 11}:
        geometry = "fluid"
    elif block_id in {18, 20, 52, 79, 138}:
        geometry = "alpha_cube"
    elif block_id in {44, 126}:
        geometry = "slab"
        height = 1.0 if metadata & 8 else 0.5
    elif block_id in {53, 67, 108, 109, 114, 128, 134, 135, 136, 156}:
        geometry = "stair"
        height = 1.0 if metadata & 4 else 0.5
    elif block_id == 78:
        geometry = "cover"
        height = ((metadata & 7) + 1) / 8
    elif block_id in {26, 70, 72, 92, 116, 118, 120, 122, 147, 148, 151, 154}:
        geometry = "partial"
        height = {26: 0.5625, 70: 0.0625, 72: 0.0625, 92: 0.5, 116: 0.75, 118: 0.3125,
                  120: 0.8125, 122: 1.0, 147: 0.0625, 148: 0.0625, 151: 0.375, 154: 0.625}.get(block_id, 1.0)
    elif block_id in {34, 36, 119}:
        geometry = "nonstandard"
    if block_id in {2, 31} and not (block_id == 31 and metadata == 0):
        tint = "grass"
    elif block_id in {18, 106}:
        tint = "pine" if block_id == 18 and metadata & 3 == 1 else "birch" if block_id == 18 and metadata & 3 == 2 else "foliage"
    elif block_id == 111:
        tint = "lily_pad"
    elif block_id in {104, 105}:
        tint = "stem"
    elif block_id in {8, 9}:
        tint = "water"
    result: dict[str, Any] = {"geometry": geometry, "height": height}
    if tint:
        result["tint"] = tint
    if block_id == 36:
        result["renderAsAir"] = True
    return result


def biome_colorizer_xy(temperature: float, rainfall: float) -> tuple[int, int]:
    temperature = max(0.0, min(1.0, temperature))
    rainfall = max(0.0, min(1.0, rainfall)) * temperature
    return int((1.0 - temperature) * 255.0), int((1.0 - rainfall) * 255.0)

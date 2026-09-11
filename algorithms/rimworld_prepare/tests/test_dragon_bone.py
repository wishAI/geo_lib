from __future__ import annotations

import copy
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

from algorithms.rimworld_prepare import builder


class DragonBoneBuildTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.mod_output = self.root / "DragonYuran"
        self.character_output = self.root / "PrepareCarefully"
        self.mod_output.joinpath("1.6", "Defs").mkdir(parents=True)
        self.patches = (
            patch.object(builder, "MOD_OUTPUT", self.mod_output),
            patch.object(builder, "CHARACTER_OUTPUT", self.character_output),
        )
        for active_patch in self.patches:
            active_patch.start()

    def tearDown(self) -> None:
        for active_patch in reversed(self.patches):
            active_patch.stop()
        self.temp.cleanup()

    @staticmethod
    def config() -> dict:
        return json.loads(builder.CONFIG.read_text(encoding="utf-8"))

    def test_framework_only_dragon_bone_defs(self) -> None:
        config = builder.validate_config(self.config())
        builder._write_about(config)
        report = builder._write_dragon_bone_defs(config)

        about = ET.parse(self.mod_output / "About" / "About.xml").getroot()
        package_ids = [node.text for node in about.findall("./modDependencies/li/packageId")]
        self.assertIn("RedMattis.BetterPrerequisites", package_ids)
        self.assertNotIn("RedMattis.BigSmall.Core", package_ids)

        defs = ET.parse(self.mod_output / "1.6" / "Defs" / "DragonYuran_DragonBone.xml").getroot()
        implant = next(node for node in defs.findall("HediffDef") if node.findtext("defName") == "DragonYuran_DragonBone")
        giant = next(node for node in defs.findall("HediffDef") if node.findtext("defName") == "DragonYuran_GiantForm")
        ability = next(node for node in defs.findall("AbilityDef") if node.findtext("defName") == "DragonYuran_GiantTransform")
        recipe = next(node for node in defs.findall("RecipeDef") if node.findtext("defName") == "InstallDragonYuranDragonBone")

        self.assertEqual(implant.findtext("./abilities/li"), "DragonYuran_GiantTransform")
        self.assertEqual(giant.findtext("./stages/li/statFactors/SM_BodySizeMultiplier"), "5.000")
        self.assertEqual(giant.find("./comps/li").attrib["Class"], "BigAndSmall.CompProperties_CanCancelHediff")
        self.assertEqual(ability.find("./comps/li").attrib["Class"], "BigAndSmall.CompProperties_AbilityGiveHediffComplex")
        self.assertEqual(recipe.findtext("./appliedOnFixedBodyParts/li"), "Spine")
        self.assertEqual(report["forbiddenGenesPackageId"], "RedMattis.BigSmall.Core")

    def test_prepare_carefully_places_bone_on_spine(self) -> None:
        character = copy.deepcopy(self.config()["characters"][0])
        character["dragonBoneInstalled"] = True
        apparel_map = {def_name: f"DragonYuran_Apparel_{def_name}" for def_name in character["apparel"]}
        output = builder._write_pcc(character, apparel_map)
        pawn = ET.parse(output).getroot().find("pawn")
        implants = pawn.findall("./implants/li")
        bone = next(node for node in implants if node.findtext("hediff") == "DragonYuran_DragonBone")
        self.assertEqual(bone.findtext("recipe"), "InstallDragonYuranDragonBone")
        self.assertEqual(bone.findtext("bodyPart"), "Spine")

    def test_giant_scale_range_is_enforced(self) -> None:
        for invalid in (1.24, 10.01, float("inf"), True):
            with self.subTest(invalid=invalid):
                config = self.config()
                config["mod"]["dragonBone"]["giantScale"] = invalid
                with self.assertRaises(ValueError):
                    builder.validate_config(config)


if __name__ == "__main__":
    unittest.main()

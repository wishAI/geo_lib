import tempfile
import unittest
from pathlib import Path

from algorithms.rimworld_prepare.character_options import collect


class CharacterOptionsTests(unittest.TestCase):
    def test_templates_resolve_xenotype_genes_and_chemical_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            defs = data / "Biotech" / "Defs"
            defs.mkdir(parents=True)
            (defs / "test.xml").write_text("""<Defs>
              <SkillDef><defName>Shooting</defName><skillLabel>shooting</skillLabel></SkillDef>
              <ChemicalDef><defName>Alcohol</defName><label>alcohol</label></ChemicalDef>
              <ChemicalDef><defName>Luciferium</defName><generateAddictionGenes>false</generateAddictionGenes></ChemicalDef>
              <GeneTemplateDef Name="AptitudeBase" Abstract="True">
                <geneTemplateType>Skill</geneTemplateType><displayCategory>Aptitudes</displayCategory>
                <exclusionTagPrefix>Aptitude</exclusionTagPrefix><biostatCpx>2</biostatCpx>
              </GeneTemplateDef>
              <GeneTemplateDef ParentName="AptitudeBase">
                <defName>AptitudeStrong</defName><label>strong {0}</label><biostatMet>-1</biostatMet>
              </GeneTemplateDef>
              <GeneTemplateDef>
                <defName>ChemicalDependency</defName><label>{0} dependency</label>
                <geneTemplateType>Chemical</geneTemplateType><exclusionTagPrefix>Addiction</exclusionTagPrefix>
                <biostatCpx>1</biostatCpx><biostatMet>4</biostatMet>
                <chemicalBiostatOverrides><li><chemical>Alcohol</chemical><biostatMet>3</biostatMet></li></chemicalBiostatOverrides>
              </GeneTemplateDef>
              <XenotypeDef><defName>Test</defName><genes>
                <li>AptitudeStrong_Shooting</li><li>ChemicalDependency_Alcohol</li>
              </genes></XenotypeDef>
              <ColorDef Name="FavoriteBase" Abstract="True"><colorType>Ideo</colorType></ColorDef>
              <ColorDef ParentName="FavoriteBase"><defName>Blue</defName><color>(0,0,1)</color></ColorDef>
              <ColorDef><defName>FloorRed</defName><colorType>Structure</colorType></ColorDef>
            </Defs>""", encoding="utf-8")
            result = collect(data)
        genes = {entry["defName"]: entry for entry in result["genes"]}
        self.assertEqual(set(genes), {"AptitudeStrong_Shooting", "ChemicalDependency_Alcohol"})
        aptitude = genes["AptitudeStrong_Shooting"]
        self.assertEqual(aptitude["label"], "strong shooting")
        self.assertEqual((aptitude["complexity"], aptitude["metabolism"]), (2, -1))
        self.assertEqual(aptitude["exclusionTags"], ["Aptitude_Shooting"])
        self.assertEqual(genes["ChemicalDependency_Alcohol"]["metabolism"], 3)
        self.assertEqual(result["unresolvedXenotypeGenes"], [])
        self.assertEqual([c["defName"] for c in result["favoriteColors"]], ["Blue"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import copy
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

from algorithms.rimworld_prepare import builder


class CharacterEditorTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(builder.CONFIG.read_text())
        self.person = self.config['characters'][0]
        self.options = {
            'hairStyles': [{'defName': 'Shaved', 'label': 'Shaved', 'asset': None}] + [
                {'defName': f'DragonYuran_Hair_{i}', 'label': f'Hair {i}', 'asset': 'AddonHair/YR_hair' + (str(i) if i else '')}
                for i in range(10)],
            'traits': [{'defName': 'Industriousness', 'degrees': [{'degree': 2}]}],
            'backstories': [{'defName': 'ChildStory', 'slot': 'Childhood'}, {'defName': 'AdultStory', 'slot': 'Adulthood'}],
            'beards': [{'defName': 'NoBeard'}],
            'tattoos': [{'defName': 'Face_Test', 'tattooType': 'Face'}, {'defName': 'Body_Test', 'tattooType': 'Body'}],
            'favoriteColors': [{'defName': 'Red'}],
            'genes': [{'defName': 'DarkVision'}, {'defName': 'AptitudeStrong_Mining'}],
            'xenotypes': [{'defName': 'Baseliner'}], 'installedExpansions': ['Core', 'Biotech'],
        }
        active = patch.object(builder, 'character_options', return_value=self.options)
        active.start()
        self.addCleanup(active.stop)

    def validate(self):
        return builder.validate_config(self.config, catalog={'characterOptions': self.options})

    def test_legacy_traits_and_hair_migrate(self):
        self.person['traits'] = [{'def': 'Industrious', 'degree': 0}]
        self.person['hairVariant'] = 7
        self.person.pop('hairDef', None)
        self.validate()
        self.assertEqual(self.person['traits'], [{'def': 'Industriousness', 'degree': 2}])
        self.assertEqual(self.person['hairDef'], 'DragonYuran_Hair_7')

    def test_all_editor_fields_export_to_native_v5_records(self):
        self.person.update(hairDef='DragonYuran_Hair_9', skinColor='#123456', hairColor='#654321',
                           childhood='ChildStory', adulthood='AdultStory', beard='NoBeard',
                           faceTattoo='Face_Test', bodyTattoo='Body_Test', favoriteColorDef='Red',
                           genes={'enabled': True, 'xenotypeDef': 'Baseliner', 'customXenotypeName': 'Custom Yuran',
                                  'endogenes': ['DarkVision'], 'xenogenes': ['AptitudeStrong_Mining']})
        apparel = self.person['apparel'][0]
        self.person['apparelDetails'] = {apparel: {'condition': 65, 'quality': 'Masterwork', 'color': '#ff8800'}}
        self.validate()
        with tempfile.TemporaryDirectory() as temp, patch.object(builder, 'CHARACTER_OUTPUT', Path(temp)):
            path = builder._write_pcc(self.person, {name: 'Test_' + name for name in self.person['apparel']})
            pawn = ET.parse(path).getroot().find('pawn')
        for field in ('hairDef', 'childhood', 'adulthood', 'beard', 'faceTattoo', 'bodyTattoo', 'favoriteColorDef'):
            self.assertEqual(pawn.findtext(field), self.person[field])
        self.assertEqual(pawn.findtext('hairColor'), builder._color_tuple('#654321'))
        self.assertEqual(pawn.findtext('genes/endogeneRecords/li/def'), 'DarkVision')
        self.assertEqual(pawn.findtext('genes/xenogeneRecords/li/def'), 'AptitudeStrong_Mining')
        self.assertEqual(pawn.findtext('apparel/li/hitPoints'), '0.65')
        self.assertEqual(pawn.findtext('apparel/li/quality'), 'Masterwork')
        self.assertEqual(pawn.findtext('traits/li/def'), 'Industriousness')

    def test_empty_gene_lists_are_explicit_and_disabled_genes_are_omitted(self):
        self.validate()
        with tempfile.TemporaryDirectory() as temp, patch.object(builder, 'CHARACTER_OUTPUT', Path(temp)):
            mapping = {name: name for name in self.person['apparel']}
            pawn = ET.parse(builder._write_pcc(self.person, mapping)).getroot().find('pawn')
            self.assertIsNone(pawn.find('genes'))
            self.person['genes']['enabled'] = True
            pawn = ET.parse(builder._write_pcc(self.person, mapping)).getroot().find('pawn')
            self.assertIsNotNone(pawn.find('genes/endogeneRecords'))
            self.assertIsNotNone(pawn.find('genes/xenogeneRecords'))
            self.assertEqual(len(pawn.find('genes/endogeneRecords')), 0)

    def test_face_tattoo_exports_body_sentinel_for_v5_loader(self):
        self.person.update(faceTattoo='Face_Test', bodyTattoo='')
        self.validate()
        with tempfile.TemporaryDirectory() as temp, patch.object(builder, 'CHARACTER_OUTPUT', Path(temp)):
            pawn = ET.parse(builder._write_pcc(self.person, {name: name for name in self.person['apparel']})).getroot().find('pawn')
        self.assertEqual(pawn.findtext('faceTattoo'), 'Face_Test')
        self.assertEqual(pawn.findtext('bodyTattoo'), 'NoTattoo_Body')

    def test_invalid_details_fail_before_export(self):
        invalid_values = [
            ('hairDef', 'Unknown'), ('hairDef', ''), ('bodyType', 'Hulk'),
            ('chronologicalAge', 13), ('childhood', 'AdultStory'), ('faceTattoo', 'Body_Test'),
            ('traits', [{'def': 'Industriousness', 'degree': 0}]),
            ('genes', {'enabled': True, 'xenotypeDef': 'Baseliner', 'endogenes': ['Unknown'], 'xenogenes': []}),
            ('genes', {'enabled': True, 'xenotypeDef': 'Baseliner', 'endogenes': ['DarkVision'], 'xenogenes': ['DarkVision']}),
            ('genes', {'enabled': True, 'xenotypeDef': 'Baseliner', 'endogenes': ['DarkVision', 'DarkVision'], 'xenogenes': []}),
            ('apparelDetails', {self.person['apparel'][0]: {'condition': 101}}),
        ]
        original = copy.deepcopy(self.person)
        for field, value in invalid_values:
            with self.subTest(field=field, value=value):
                self.person.clear()
                self.person.update(copy.deepcopy(original))
                self.person[field] = value
                with self.assertRaises(ValueError):
                    self.validate()

    def test_generated_race_uses_persistent_hair_defs_without_forced_hair_addon(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(builder, 'MOD_OUTPUT', Path(temp)):
            builder._write_race_defs()
            defs = ET.parse(Path(temp) / '1.6/Defs/DragonYuran_Race.xml').getroot()
        self.assertEqual(len(defs.findall('HairDef')), 10)
        self.assertEqual(defs.findtext('HairDef/texPath'), 'DragonYuran/AddonHair/YR_hair')
        race = defs.find('AlienRace.ThingDef_AlienRace')
        paths = [node.text for node in race.findall('alienRace/generalSettings/alienPartGenerator/bodyAddons/li/path')]
        self.assertEqual(len(paths), 3)
        self.assertFalse(any('AddonHair' in path for path in paths))
        self.assertEqual(race.findtext('alienRace/styleSettings/li/value/styleTagsOverride/li'), 'DragonYuranHair')


if __name__ == '__main__':
    unittest.main()

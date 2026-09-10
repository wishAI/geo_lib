import copy
import importlib.util
import json
import re
import sqlite3
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from PIL import Image

SOURCE=Path(__file__).resolve().parents[1]/'mod_builder.py'
spec=importlib.util.spec_from_file_location('eqn_builder',SOURCE)
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)

class ModCompilerTests(unittest.TestCase):
    def setUp(self):self.d=json.loads(builder.CONFIG.read_text())
    def test_rejects_cycles_nonfinite_and_asset_escape(self):
        for mutate in (
            lambda d:d['mod']['technologies'][0].update(prerequisite=d['mod']['technologies'][-1]['id']),
            lambda d:d['mod']['technologies'][0].update(weaponCooldown=0),
            lambda d:d['mod']['technologies'][0].update(hull=float('inf')),
            lambda d:d['mod']['species'][0]['portraits'][0].update(texture='../README.md'),
            lambda d:d['mod']['ship'].update(maxBodies=4),
            lambda d:d['mod']['eventIdeas'][0].update(implemented=True),
            lambda d:d['ship']['materialGroups']['body'].update(opacity=1.1),
            lambda d:d['ship']['materialGroups']['eyes'].update(crystal='yes'),
            lambda d:d['animation']['fastFlightSync'].update(wingCycles=0),
            lambda d:d['animation']['fastFlightSync'].update(bodyDuration=0),
            lambda d:d['animation']['fastFlightSync'].update(wingSpeed=float('nan')),
            lambda d:d['animation']['fastFlightSync'].update(tailPhase=1.1),
            lambda d:d['animation']['fastFlightSync'].update(lockSeamlessLoop='yes'),
        ):
            d=copy.deepcopy(self.d);mutate(d)
            with self.assertRaises((ValueError,KeyError)):builder.validate_config(d)
    def test_generated_package_preserves_guards_and_exports_edits(self):
        self.d['mod']['ground']['damage']=321
        self.d['mod']['leader']['traits'][0]['weaponDamage']=.42
        self.d['mod']['technologies'][-1]['weaponDamage']=1234567
        self.d['mod']['technologies'][-1]['groundMultiplier']=777
        self.d['mod']['species'][0]['portraits'][0]['enabled']=False
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);builder.Compiler(builder.validate_config(self.d),out).build()
            report=builder.validate_export(out,self.d)
            self.assertEqual(report['species'][0]['portraits'],7)
            self.assertEqual(report['weaponSlots'],65)
            self.assertFalse(report['gameLoadVerified'])
            self.assertFalse(report['militaryPowerEstimator']['combatStatsChanged'])
            self.assertEqual(len(report['militaryPowerEstimator']['stages']),11)
            self.assertAlmostEqual(report['militaryPowerEstimator']['stages'][-1]['estimatorWeaponDamage'],18)
            species=(out/'common/species_classes/eqn_species.txt').read_text()
            self.assertIn('playable = { always = yes } randomized = no',species)
            self.assertNotIn('\nROBOT =',species);self.assertNotIn('\nPLANT =',species)
            decisions=(out/'common/decisions/eqn_bodies.txt').read_text()
            for n in range(1,4):
                self.assertIn(f'any_owned_ship = {{ has_ship_flag = eqn_body_{n}',decisions)
                self.assertIn(f'any_owned_army = {{ army_type = eqn_ground_{n}',decisions)
            self.assertIn('damage = 321',(out/'common/armies/eqn_ground.txt').read_text())
            self.assertIn('ship_weapon_damage = 0.42',(out/'common/traits/eqn_leaders.txt').read_text())
            self.assertIn('leader_class = { scientist official commander }',(out/'common/traits/eqn_leaders.txt').read_text())
            self.assertIn('all_technology_research_speed = 0.25',(out/'common/traits/eqn_leaders.txt').read_text())
            self.assertIn('country_unity_produces_mult = 0.25',(out/'common/traits/eqn_leaders.txt').read_text())
            self.assertEqual((out/'common/traits/eqn_leaders.txt').read_text().count('triggered_army_modifier'),1)
            self.assertIn('mult = value:eqn_ground_progression',(out/'common/traits/eqn_leaders.txt').read_text())
            progression=(out/'common/script_values/eqn_ground.txt').read_text()
            self.assertIn('add = 776',progression)
            self.assertIn('has_technology = tech_eqn_eternal',progression)
            self.assertIn('min = 1234567',(out/'common/component_templates/eqn_components.txt').read_text())
            self.assertIn('class = shipclass_military_special',(out/'common/ship_sizes/eqn_heart.txt').read_text())
            anim=(out/'gfx/models/ships/stellaris_heart/stellaris_heart.asset').read_text()
            self.assertIn('file = "heart_idle.anim"',anim)
            self.assertIn('file = "heart_planet_killer.anim"',anim)
            self.assertIn('name = "working" animation = "planet_killer"',anim)
            self.assertNotIn('file = "gfx/',anim)
            self.assertEqual(self.d['animation']['fastFlightSync'], {
                'bodyDuration': 4.0, 'bodySpeed': 1.0,
                'wingCycles': 7.0, 'wingSpeed': 1.0, 'wingPhase': 0.0,
                'tailCycles': 4.0, 'tailPhase': 0.0,
                'lockSeamlessLoop': True,
            })
            component=(out/'common/component_templates/eqn_components.txt').read_text()
            self.assertEqual(component.count('military_power_multiplier ='),77)
            self.assertIn('military_power_multiplier = 1\n',component)
            self.assertIn('military_power_multiplier = 0.00001458\n',component)
            self.assertIn('key = "EQN_HORN_W" size = planet_killer type = planet_killer',component)
            self.assertIn('windup = { min = 1 max = 1 }',component)
            self.assertIn('total_fire_time = 15 accuracy = 1',component)
            planet=(out/'gfx/projectiles/planet_destruction/eqn_planet_destruction.txt').read_text()
            self.assertIn('windup = { duration = 1 }',planet)
            self.assertIn('start = { duration = 2 ',planet)
            self.assertIn('in_progress = { duration = 10 ',planet)
            self.assertIn('end = { duration = 2 ',planet)
            self.assertIn('texture_scroll_speed = { 0.0 1.5 }',planet)
            trigger=(out/'common/scripted_triggers/eqn_planet_killer.txt').read_text()
            self.assertIn('can_destroy_planet_with_EQN_HORN_W',trigger)
            self.assertIn('can_destroy_planet_with_PLANET_KILLER_CRACKER = yes',trigger)
            action=(out/'common/on_actions/eqn_planet_killer.txt').read_text()
            self.assertIn('on_destroy_planet_with_EQN_HORN_W',action)
            self.assertIn('planet_destruction.100',action)
            start_actions=(out/'common/on_actions/eqn_actions.txt').read_text()
            self.assertIn('on_ruler_set = { events = { eqn_go.2 } }',start_actions)
            self.assertIn('on_single_player_save_game_load = { events = { eqn_go.4 } }',start_actions)
            events=(out/'events/eqn_mechanical_events.txt').read_text()
            self.assertIn('change_leader_portrait = eqn_heart_portrait',events)
            self.assertIn('add_trait = { trait = eqn_princess show_message = no }',events)
            self.assertNotIn('add_trait = { trait = eqn_battle_harmony',events)
            self.assertIn('event = { id = eqn_go.4',events)
            mechanics=(out/'common/scripted_effects/eqn_mechanics.txt').read_text()
            self.assertEqual(mechanics.count('NOT = { exists = event_target:eqn_body_location }'),3)
            loc=(out/'localisation/english/eqn_go_l_english.yml').read_text(encoding='utf-8-sig')
            self.assertIn('EQN_HORN_W_ACTION',loc)
            self.assertIn('FLEETORDER_DESTROY_PLANET_WITH_EQN_HORN_W',loc)
            self.assertTrue((out/'common/name_lists/eqn_pony.txt').read_bytes().startswith(b'\xef\xbb\xbf'))
            descriptor=(out/'descriptor.mod').read_text()
            self.assertIn('supported_version = "v4.4.*"',descriptor)
            self.assertIn('picture = "thumbnail.png"',descriptor)
            self.assertTrue((out/'thumbnail.png').is_file())
            with Image.open(out/'thumbnail.png') as thumbnail:
                self.assertEqual(thumbnail.size,(256,256))
            self.assertNotIn('wormhole_station',(out/'common/name_lists/eqn_pony.txt').read_text())
            for rel,expected in {
                'gfx/interface/icons/eqn_research.dds':(58,58),
                'gfx/interface/icons/eqn_ground.dds':(34,34),
                'gfx/interface/icons/eqn_heart.dds':(29,29),
                'gfx/interface/icons/eqn_trait_research.dds':(29,29),
                'gfx/interface/icons/eqn_origin.dds':(40,40),
                'gfx/interface/icons/technologies/tech_guardian.dds':(52,52),
                'gfx/event_pictures/eqn_origin.dds':(220,115),
                'gfx/portraits/eqn_heart.dds':(300,256),
                'gfx/interface/icons/ship_parts/eqn_component_horn_w.dds':(58,58),
                'gfx/interface/icons/technologies/tech_ascendant.dds':(52,52),
            }.items():
                header=(out/rel).read_bytes()[:20]
                self.assertEqual((int.from_bytes(header[16:20],'little'),int.from_bytes(header[12:16],'little')),expected,rel)
    def test_standalone_has_no_acot_dependency_or_boss_reward_gate(self):
        self.d['mod']['compatibility']['mode']='standalone'
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);builder.Compiler(self.d,out).build()
            self.assertNotIn('dependencies',(out/'descriptor.mod').read_text())
            tech=(out/'common/technology/eqn_technologies.txt').read_text()
            self.assertNotIn('tech_dark_matter_',tech)
            self.assertIn('tech_eqn_eternal',tech)
    def test_linux_install_registers_local_launcher_thumbnail_with_backup(self):
        with tempfile.TemporaryDirectory() as folder:
            home=Path(folder);user=home/'.local/share/Paradox Interactive/Stellaris';mods=user/'mod';mods.mkdir(parents=True)
            previous=mods/'eqn_go';previous.mkdir();(previous/'old.txt').write_text('0.1.7')
            source=home/'build';source.mkdir();(source/'descriptor.mod').write_text('name = "EQN-GO"\n');(source/'thumbnail.png').write_bytes(b'png')
            database=user/'launcher-v2.sqlite'
            with sqlite3.connect(database) as connection:
                connection.execute('CREATE TABLE mods (source TEXT, gameRegistryId TEXT, dirPath TEXT, thumbnailPath TEXT)')
                connection.execute("INSERT INTO mods VALUES ('local','mod/eqn_go.mod',?,NULL)",(str(previous),))
            with mock.patch.object(builder.sys,'platform','linux'),mock.patch.object(builder.Path,'home',return_value=home):
                report=builder.install(source)
            self.assertEqual((mods/'eqn_go.previous/old.txt').read_text(),'0.1.7')
            self.assertEqual((mods/'eqn_go/thumbnail.png').read_bytes(),b'png')
            with sqlite3.connect(database) as connection:
                thumbnail=connection.execute('SELECT thumbnailPath FROM mods').fetchone()[0]
            self.assertEqual(thumbnail,str(mods/'eqn_go/thumbnail.png'))
            self.assertEqual(report['launcherThumbnail']['rowsUpdated'],1)
            self.assertTrue((user/'launcher-v2.pre-eqn-local-thumbnail.sqlite').is_file())
    def test_parser_rejects_unterminated_scripts(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'bad.txt';p.write_text('name = { text = "bad }')
            with self.assertRaises(ValueError):builder.parse_check(p)
    def test_component_icon_rejects_oversized_master(self):
        icon=self.d['mod']['artwork']['icons'][1]
        icon['nativeTextures']['component']=icon['texture']
        with self.assertRaisesRegex(ValueError,'component artwork must be 58 square'):
            builder.validate_config(self.d)
    def test_generated_icons_keep_transparent_edges_and_white_foreground_detail(self):
        icons=self.d['mod']['artwork']['componentIcons']+self.d['mod']['artwork']['technologyIcons']
        for icon in icons:
            image=Image.open(builder.asset(icon['preview'])).convert('RGBA')
            self.assertTrue(all(image.getpixel(point)[3]==0 for point in ((0,0),(255,0),(0,255),(255,255))),icon['id'])
            pixels=image.get_flattened_data() if hasattr(image,'get_flattened_data') else image.getdata()
            white=sum(1 for red,green,blue,alpha in pixels if alpha>128 and min(red,green,blue)>220 and max(red,green,blue)-min(red,green,blue)<30)
            self.assertGreater(white,100,icon['id'])
    def test_two_species_choices_contain_all_portrait_variants(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);builder.Compiler(self.d,out).species()
            sets=(out/'common/portrait_sets/eqn_portraits.txt').read_text()
            visible=re.findall(r'species_class = (\w+) portraits = \{ ([^}]+) \}',sets)
            self.assertEqual(visible,[('EQN_EEVEE','eqn_eevee_group'),('EQN_KRYS','eqn_krys_group')])
            portraits=(out/'gfx/portraits/portraits/eqn_portraits.txt').read_text()
            for species in self.d['mod']['species']:
                group=species['id'].lower()+'_group'
                body=portraits.split(group+' = {',1)[1].split('\n',1)[0]
                for context in ('game_setup','species','pop','leader','ruler'):
                    pool=re.search(context+r' = \{ add = \{ portraits = \{ ([^}]+)',body)[1].split()
                    self.assertEqual(pool,[p['id'] for p in species['portraits'] if p['enabled']])
            self.assertIn('eqn_krys21 = { texturefile =',portraits)
            self.assertEqual(sets.count('playable = { always = no } randomizable = { always = no }'),2)
            self.assertNotRegex(sets,r'\b(?:playable|randomizable)\s*=\s*(?:yes|no)\b')
    def test_bilingual_keys_stay_stable_when_portraits_change(self):
        self.d['mod']['species'][0]['portraits'][0]['enabled']=False
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);compiler=builder.Compiler(self.d,out);compiler.build()
            translations=self.d['mod']['localisation']['translations']
            self.assertFalse(set(compiler.loc)-set(translations))
            localized={}
            for language in ('english','simp_chinese'):
                raw=(out/f'localisation/{language}/eqn_go_l_{language}.yml').read_bytes()
                self.assertTrue(raw.startswith(b'\xef\xbb\xbf'))
                localized[language]=dict(re.findall(r'^ (\w+):0 "(.*)"$',raw.decode('utf-8-sig'),re.M))
            self.assertEqual(localized['english'].keys(),localized['simp_chinese'].keys())
            self.assertEqual(localized['simp_chinese']['eqn_commander_name_1'],'星海公主·化身1')
            self.assertIn('eqn_name_sequence_0',localized['simp_chinese'])
            self.assertNotEqual(localized['english']['eqn_ground_1'],localized['english']['eqn_deploy_ground_1'])
            decisions=(out/'common/decisions/eqn_bodies.txt').read_text()
            self.assertEqual(decisions.count('alloys = 10000'),3)
            self.assertEqual(decisions.count('potential = { is_capital = yes'),3)
            self.assertEqual(decisions.count('potential = { is_colony = yes'),6)
            self.assertEqual(decisions.count('save_event_target_as = eqn_body_location owner = { every_owned_army'),3)
            english=(out/'localisation/english/eqn_go_l_english.yml').read_text(encoding='utf-8-sig')
            self.assertIn('right-click a hostile planet to land',english)

if __name__=='__main__':unittest.main()

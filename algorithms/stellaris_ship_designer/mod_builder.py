"""EQN-GO compiler. The editable JSON is the sole gameplay source of truth.

Uses only the standard library. Mechanical events implement requested origin,
body switching and technology upgrades; optional narrative events are not emitted.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / 'inputs/stellaris_heart.json'
IDENT = re.compile(r'^[A-Za-z][A-Za-z0-9_]{0,79}$')
SIZES = {'S':'small','M':'medium','L':'large','X':'extra_large','T':'titanic','P':'point_defence','G':'torpedo'}
FACTORS = {'S':1,'M':1.5,'L':2,'X':3,'T':5,'P':.6,'G':2}
ICON_SIZES = {'component':58, 'army':34, 'trait':29, 'origin':40, 'technology':52}

def quote(value):
    return '"' + str(value).replace('\\','\\\\').replace('"','\\"').replace('\n','\\n').replace('\r','') + '"'

def number(value):
    return format(float(value), '.12g')

def asset(relative):
    path = (ROOT / relative).resolve()
    path.relative_to((ROOT / 'inputs/eqn_assets').resolve())
    if not path.is_file():
        raise ValueError(f'Missing EQN asset: {relative}')
    return path

def icon_texture(icon, usage):
    fallback=str(Path(icon['texture']).with_name(Path(icon['texture']).stem+'_'+usage+'.dds'))
    path=asset(icon.get('nativeTextures',{}).get(usage,fallback))
    header=path.read_bytes()[:20]
    dimensions=(int.from_bytes(header[16:20],'little'),int.from_bytes(header[12:16],'little'))
    if header[:4]!=b'DDS ' or dimensions!=(ICON_SIZES[usage],)*2:
        raise ValueError(f'{usage} artwork must be {ICON_SIZES[usage]} square: {path.name}')
    return path

def fixed_texture(relative, size, usage):
    path=asset(relative);data=path.read_bytes();header=data[:20]
    dimensions=(int.from_bytes(header[16:20],'little'),int.from_bytes(header[12:16],'little'))
    if header[:4]!=b'DDS ' or dimensions!=(size,size):
        raise ValueError(f'{usage} artwork must be {size} square: {path.name}')
    # Generated icons are uncompressed 32-bit RGBA DDS. Transparent corners are
    # part of the contract; this catches accidentally baked checkerboards or
    # opaque black/white margins before a package reaches the game.
    if len(data)!=128+size*size*4 or any(data[128+(y*size+x)*4+3] for x,y in ((0,0),(size-1,0),(0,size-1),(size-1,size-1))):
        raise ValueError(f'{usage} artwork must have a transparent background: {path.name}')
    if not any(data[131::4]):
        raise ValueError(f'{usage} artwork has no visible foreground: {path.name}')
    return path

def validate_config(d):
    if not isinstance(d,dict) or not isinstance(d.get('mod'),dict):
        raise ValueError('EQN-GO requires a mod object')
    m=d['mod']
    if m.get('id')!='eqn_go' or m.get('schemaVersion')!=1:
        raise ValueError('Unsupported mod id or schema')
    if m['ship']['maxBodies'] != 3:
        raise ValueError('The acceptance contract allows exactly three body slots')
    if m['compatibility']['mode'] not in ('acot','standalone'):
        raise ValueError('Compatibility mode must be acot or standalone')
    if m['ship']['behavior'] not in ('swarm','picket','line','artillery','carrier'):
        raise ValueError('Unknown combat behavior')
    if not 0.25<=d['ship'].get('crystalOpacity',1)<=1:
        raise ValueError('Crystal opacity must be between 0.25 and 1')
    if not isinstance(d['ship'].get('hideMouthInterior'),bool):
        raise ValueError('Hide mouth interior must be true or false')
    groups=d['ship'].get('materialGroups',{})
    if set(groups)!={'gold','wings','eyes','body','maneTail'}:
        raise ValueError('Provide the five Heart material groups')
    for key,group in groups.items():
        if not isinstance(group.get('label'),str) or not group['label'].strip():
            raise ValueError(f'Material group {key} needs a label')
        if not isinstance(group.get('crystal'),bool) or not isinstance(group.get('faceted'),bool):
            raise ValueError(f'Material group {key} toggles must be true or false')
        for prop in ('opacity','metallic','roughness','clearcoat','transmission','shellOpacity'):
            value=group.get(prop)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:
                raise ValueError(f'Material group {key}.{prop} must be between 0 and 1')
    sync=d['animation'].get('fastFlightSync',{})
    for key,minimum,maximum in (
        ('bodyDuration',.5,12), ('bodySpeed',.1,4),
        ('wingCycles',.25,12), ('wingSpeed',.1,4), ('tailCycles',.25,12),
    ):
        value=sync.get(key)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not minimum<=value<=maximum:
            raise ValueError(f'Fast flight {key} must be between {minimum} and {maximum}')
    for key in ('wingPhase','tailPhase'):
        phase=sync.get(key)
        if isinstance(phase,bool) or not isinstance(phase,(int,float)) or not math.isfinite(phase) or not 0<=phase<=1:
            raise ValueError(f'Fast flight {key} must be between 0 and 1')
    if not isinstance(sync.get('lockSeamlessLoop'),bool):
        raise ValueError('Fast flight lockSeamlessLoop must be true or false')
    if not 1<=m['ship'].get('planetKillerSpeedMultiplier',1)<=20:
        raise ValueError('Planet-killer speed multiplier must be between 1 and 20')
    if not -90<=d['animation'].get('planetKillerPitch',-45)<=90:
        raise ValueError('Planet-killer pitch must be between -90 and 90 degrees')
    seen=set()
    def ident(v):
        if not isinstance(v,str) or not IDENT.fullmatch(v) or v in seen:
            raise ValueError(f'Invalid or duplicate script identifier: {v}')
        seen.add(v)
    def finite(v,path):
        if isinstance(v,bool):return
        if isinstance(v,(int,float)) and (not math.isfinite(v) or not 0<=v<=1e12):
            raise ValueError(f'{path} must be finite, non-negative and at most 1e12')
        if isinstance(v,dict):
            for k,x in v.items():finite(x,path+'.'+k)
        if isinstance(v,list):
            for i,x in enumerate(v):finite(x,path+f'.{i}')
    finite(m,'mod')
    for key in ('name','version','supportedVersion','description'):
        if not isinstance(m.get(key),str) or not m[key].strip():raise ValueError(f'Missing mod {key}')
    if not re.fullmatch(r'v?\d+\.\d+\.(?:\d+|\*)',m['supportedVersion']):raise ValueError('Supported version must look like v4.4.*')
    ident(m['ship']['size']);ident(m['origin']['id'])
    if not m['ship']['size'].startswith('eqn_') or not m['origin']['id'].startswith('origin_eqn_'):raise ValueError('Ship and origin identifiers must remain namespaced')
    for key,value in m.get('localisation',{}).get('translations',{}).items():
        if not IDENT.fullmatch(key) or not isinstance(value,str):raise ValueError('Invalid Chinese localisation entry')
    if not 1<=m['origin']['startingLeaderLevel']<=10:raise ValueError('Leader level must be 1–10')
    if not 1<=len(m['technologies'])<=30:raise ValueError('Provide 1–30 technologies')
    previous=set()
    for t in m['technologies']:
        ident(t['id'])
        if not t['id'].startswith('tech_eqn_'):raise ValueError('Technology IDs must use the tech_eqn_ namespace')
        if t['area'] not in ('physics','society','engineering') or not 1<=t['tier']<=5:raise ValueError('Technology area/tier is invalid')
        if t['prerequisite'] and t['prerequisite'] not in previous:raise ValueError('Technology prerequisites must reference an earlier technology')
        if t['externalPrerequisite'] and not IDENT.fullmatch(t['externalPrerequisite']):raise ValueError('Invalid external technology id')
        if t['weaponCooldown']<=0 or t['hull']<=0 or t['cost']<=0:raise ValueError('Technology cooldown, hull and cost must be positive')
        previous.add(t['id'])
    if m['ship']['weaponCooldown']<=0:raise ValueError('Awakening weapon cooldown must be positive')
    if [t['body'] for t in m['bodyTechnologies']] != [2,3]:raise ValueError('Provide body technologies 2 and 3')
    for t in m['bodyTechnologies']:
        ident(t['id'])
        if t['id'] != f"tech_eqn_body_{t['body']}" or t['prerequisite'] not in previous or t['cost']<=0 or not 1<=t['tier']<=5:raise ValueError('Invalid body technology')
        previous.add(t['id'])
    for s in m['species']:
        ident(s['id'])
        ident(s['id'].lower()+'_group')
        if not s['id'].startswith('EQN_'):raise ValueError('Species IDs must use the EQN_ namespace; vanilla classes cannot be overwritten')
        if s['archetype'] not in ('BIOLOGICAL','MACHINE','ROBOT'):raise ValueError('Unsupported species archetype')
        if not IDENT.fullmatch(s['graphicalCulture']):raise ValueError('Invalid graphical culture')
        if not any(p['enabled'] for p in s['portraits']):raise ValueError('Each species needs at least one enabled portrait')
        for p in s['portraits']:
            ident(p['id']);asset(p['texture']);asset(p['preview'])
    for icon in m['artwork']['icons']:
        asset(icon['texture']);asset(icon['preview'])
        for usage in ICON_SIZES:icon_texture(icon,usage)
    component_ids={'horn_s','horn_m','horn_l','horn_x','horn_t','horn_p','horn_g','horn_w','crystal_core','royal_mind','crystal_wings'}
    component_icons={icon['id']:icon for icon in m['artwork']['componentIcons']}
    if set(component_icons)!=component_ids:raise ValueError('Provide one distinct icon for every EQN ship component family')
    for icon in component_icons.values():asset(icon['preview']);fixed_texture(icon['texture'],58,'component')
    technology_ids={'tech_guardian','tech_crown','tech_sovereign','tech_ascendant','tech_delta','tech_alpha','tech_sigma','tech_phi','tech_omega','tech_eternal','tech_body_2','tech_body_3'}
    technology_icons={icon['id']:icon for icon in m['artwork']['technologyIcons']}
    if set(technology_icons)!=technology_ids:raise ValueError('Provide one distinct icon for every EQN technology')
    for icon in technology_icons.values():asset(icon['preview']);fixed_texture(icon['texture'],52,'technology')
    if len({hashlib.sha256(asset(icon['texture']).read_bytes()).hexdigest() for icon in [*component_icons.values(),*technology_icons.values()]}) != len(component_icons)+len(technology_icons):
        raise ValueError('Every component and technology icon must be visually distinct')
    for tech in [*m['technologies'],*m['bodyTechnologies']]:
        if tech.get('icon') not in technology_icons:raise ValueError(f'Unknown technology icon: {tech.get("icon")}')
    asset(m['leader']['portrait']);asset(m['leader']['portraitPreview'])
    asset(m['artwork']['originPicture']);asset(m['artwork']['originPreview']);asset(m['ground']['icon'])
    if set(m['leader'].get('rulerBonuses',{}))!={'researchSpeed','unity','navalCapacity','shipBuildSpeed'}:raise ValueError('Provide all four Heart ruler bonuses')
    for t in m['leader']['traits']:
        ident(t['id'])
        if t.get('iconStyle','crystal_pony') not in ('crystal_pony','crystal_research'):raise ValueError('Unknown leader icon style')
        if not t['id'].startswith('eqn_'):raise ValueError('Leader trait IDs must use the eqn_ namespace')
    if not m['leader']['traits']:raise ValueError('Heart needs at least one leader trait')
    if any(e.get('implemented') for e in m['eventIdeas']):raise ValueError('Narrative events must remain proposals')
    slots=[s for sec in d['sections'] for s in sec.get('slots',[]) if s.get('enabled',True)]
    if not 1<=len(slots)<=256:raise ValueError('Export supports 1–256 enabled slots')
    for s in slots:
        ident(s['id'])
        if s['type'] in ('utility','auxiliary'):
            if s['size'] not in (('A',) if s['type']=='auxiliary' else ('S','M','L')):raise ValueError('Invalid utility slot size')
        elif s['size']=='W':
            if s['type']!='planet_killer':raise ValueError('W slots must be planet killers')
        elif s['size'] not in SIZES:raise ValueError('Military export supports S/M/L/X/T/P/G/W weapons; H needs another validated hull integration')
        if s['type'] not in ('utility','auxiliary') and s.get('sourceTemplateLocator','horn_muzzle')!='horn_muzzle':raise ValueError('Native Heart weapons must bind to the exported horn_muzzle')
    if sum(s.get('enabled',True) and s.get('size')=='W' for sec in d['sections'] for s in sec.get('slots',[])) != 1:
        raise ValueError('Heart requires exactly one enabled W planet-killer slot')
    for k in ('hullPoints','combatSpeed','fleetSize','entityScale','evasion'):
        v=d['ship'][k]
        if not isinstance(v,(int,float)) or not math.isfinite(v) or v<0:raise ValueError(f'Invalid ship {k}')
    if d['ship']['hullPoints']<=0 or d['ship']['entityScale']<=0:raise ValueError('Hull and scale must be positive')
    if not 0<=d['ship']['evasion']<=90:raise ValueError('Ship evasion must be 0–90 percent')
    return d

class Compiler:
    def __init__(self,d,out):
        self.d=d;self.m=d['mod'];self.out=out;self.loc={};self.files=[]
        self.size=self.m['ship']['size'];self.origin=self.m['origin']['id']
        self.component_icons={icon['id']:icon for icon in self.m['artwork']['componentIcons']}
        self.technology_icons={icon['id']:icon for icon in self.m['artwork']['technologyIcons']}
        self.stages=[dict(id='',name='Awakening',hull=d['ship']['hullPoints'],armor=self.m['ship']['armor'],shield=self.m['ship']['shield'],regen=self.m['ship']['hullRegen'],hullRegen=self.m['ship']['hullRegen'],armorRegen=self.m['ship']['armorRegen'],shieldRegen=self.m['ship']['shieldRegen'],weaponDamage=self.m['ship']['weaponDamage'],weaponCooldown=self.m['ship']['weaponCooldown'],weaponRange=self.m['ship']['weaponRange'],groundMultiplier=1)]+self.m['technologies']
    def write(self,path,text):
        p=self.out/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text.strip()+'\n',encoding='utf-8');self.files.append(path)
    def localized(self,key,name,desc=None):
        self.loc[key]=name
        if desc is not None:self.loc[key+'_desc']=desc
        return key
    def restriction(self):return f'potential = {{ is_ship_size = {self.size} }}'
    def build(self):
        self.species();self.hero();self.components();self.technology();self.leaders();self.mechanics();self.graphics()
        for lang in ('english','simp_chinese'):
            # All generated keys resolve in either UI language; editable source strings are preserved.
            translations=self.m.get('localisation',{}).get('translations',{}) if lang=='simp_chinese' else {}
            text='l_'+lang+':\n'+'\n'.join(' '+k+':0 '+quote(translations.get(k,v)) for k,v in self.loc.items())+'\n'
            p=self.out/f'localisation/{lang}/eqn_go_l_{lang}.yml';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text,encoding='utf-8-sig')
        deps=''
        if self.m['compatibility']['mode']=='acot':
            deps='dependencies = { "Ancient Cache of Technologies" "Ancient Cache of Technologies : Secrets Beyond The Gates" "Ancient Cache of Technologies: Override" }'
        self.write('descriptor.mod',f'name = {quote(self.m["name"])}\nversion = {quote(self.m["version"])}\nsupported_version = {quote(self.m["supportedVersion"])}\npicture = "thumbnail.png"\ntags = {{ "Species" "Origins" "Ships" "Technologies" }}\n{deps}')
        launcher=next(icon for icon in self.m['artwork']['icons'] if icon['id']=='crystal_pony')
        shutil.copy2(asset(launcher['preview']),self.out/'thumbnail.png')
    def species(self):
        classes=[];sets=[];cats=[];portraits=[];groups=[]
        for s in self.m['species']:
            ids=[];self.localized(s['id'],s['name']);self.localized(s['id']+'_desc',s['name']+' — player-created empires only.')
            for p in s['portraits']:
                if not p['enabled']:continue
                dst=f'gfx/models/portraits/eqn_go/{s["id"]}/{Path(p["texture"]).name}'
                target=self.out/dst;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(asset(p['texture']),target)
                portraits.append(f'{p["id"]} = {{ texturefile = {quote(dst)} }}');ids.append(p['id']);self.localized(p['id'],p['name'])
            trait='trait_organic' if s['archetype']=='BIOLOGICAL' else 'trait_machine_unit' if s['archetype']=='MACHINE' else 'trait_mechanical'
            classes.append(f'{s["id"]} = {{ archetype = {s["archetype"]} trait = {trait} graphical_culture = {s["graphicalCulture"]} portrait_modding = yes playable = {{ always = yes }} randomized = no }}')
            joined=' '.join(ids)
            group=s['id'].lower()+'_group'
            default=s.get('defaultPortrait') if s.get('defaultPortrait') in ids else ids[0]
            contexts=' '.join(f'{context} = {{ add = {{ portraits = {{ {joined} }} }} }}' for context in ('game_setup','species','pop','leader','ruler'))
            groups.append(f'{group} = {{ default = {default} {contexts} }}')
            self.localized(group,s['name'])
            # The chooser exposes one group per species. Hidden leaf membership
            # preserves the class association of already-created empire presets.
            sets.append(f'{s["id"]}_portraits = {{ species_class = {s["id"]} portraits = {{ {group} }} non_randomized_portraits = {{ {group} {joined} }} non_pre_ftl_portraits = {{ {group} {joined} }} conditional_portraits = {{ playable = {{ always = no }} randomizable = {{ always = no }} portraits = {{ {joined} }} }} }}')
            cats.append(f'{s["id"]}_category = {{ name = {s["id"]} sets = {{ {s["id"]}_portraits }} }}')
        self.write('common/species_classes/eqn_species.txt','\n'.join(classes))
        self.write('common/portrait_sets/eqn_portraits.txt','\n'.join(sets));self.write('common/portrait_categories/eqn_portraits.txt','\n'.join(cats))
        self.write('gfx/portraits/portraits/eqn_portraits.txt','portraits = {\n'+'\n'.join(portraits)+'\n eqn_heart_portrait = { texturefile = "gfx/portraits/eqn_heart.dds" }\n}\nportrait_groups = {\n'+'\n'.join(groups)+'\n}')
        flags=ROOT/'inputs/eqn_assets/flags'
        if flags.exists():shutil.copytree(flags,self.out/'flags',dirs_exist_ok=True)
        names=(ROOT/'inputs/eqn_assets/Pony.txt').read_text(encoding='utf-8-sig')
        names=re.sub(r'(?m)^Pony\s*=', 'EQN_Pony =',names,count=1)
        names=re.sub(r'(?m)#.*$', '', names)
        names=re.sub(r'(?ms)^\s*(wormhole_station|outpost_station)\s*=\s*\{[^}]*\}', '', names)
        sequence_index=0
        def sequential(match):
            nonlocal sequence_index
            key='eqn_name_sequence_'+str(sequence_index)
            sequence_index+=1
            self.localized(key,match[1])
            return 'sequential_name = '+quote(key)
        names=re.sub(r'sequential_name\s*=\s*"([^"]+)"',sequential,names)
        self.write('common/name_lists/eqn_pony.txt',names)
        namepath=self.out/'common/name_lists/eqn_pony.txt'
        namepath.write_text(namepath.read_text(),encoding='utf-8-sig')
        self.localized('name_list_EQN_Pony','EQN Pony')
    def hero(self):
        m=self.m;s=m['ship'];ship=self.d['ship'];size=self.size
        self.localized(size,ship['name']);self.localized(size+'_plural',ship['name']+' Bodies')
        self.localized(self.origin,m['origin']['name'],m['origin']['description']);self.localized('eqn_origin_effects',m['origin']['description'])
        self.write('common/governments/civics/eqn_origin.txt',f'''{self.origin} = {{
 is_origin = yes icon = "gfx/interface/icons/eqn_origin.dds" picture = GFX_eqn_origin
 potential = {{ always = yes }} possible = {{ }}
 ai_playable = {{ always = no }} random_weight = {{ base = 0 }}
 description = eqn_origin_effects
}}''')
        self.write('common/ship_sizes/eqn_heart.txt',f'''{size} = {{
 entity = eqn_heart_frame_entity
 use_shipnames_from = corvette
 formation_priority = 1 max_speed = {number(ship['combatSpeed'])} acceleration = 0.5 rotation_speed = 0.3
 collision_radius = 2 max_hitpoints = {number(ship['hullPoints'])}
 modifier = {{ ship_evasion_add = {number(ship['evasion'])} }}
 size_multiplier = 1 fleet_slot_size = {number(ship['fleetSize'])} combat_size_multiplier = 1
 map_counter_icon = ship_counter_4 icon = ship_size_military_1
 section_slots = {{ "mid" = {{ locator = "part1" }} }} num_target_locators = 0
 is_space_station = no base_buildtime = {number(s['buildDays'])}
 can_have_federation_design = no enable_default_design = yes enable_3dview_in_ship_browser = yes
 default_behavior = {s['behavior']} class = shipclass_military_special construction_type = starbase_shipyard
 potential_country = {{ is_ai = no has_origin = {self.origin} }}
 potential_construction = {{ always = no }}
 hero_ship = {{ capabilities = {{ can_repair can_upgrade }} }} auto_upgrade = yes
 required_component_set = "eqn_crystal_core"
 required_component_set = "eqn_royal_mind"
 required_component_set = "ftl_components"
 required_component_set = "eqn_crystal_wings"
 required_component_set = "EQN_HORN_W"
 required_component_set = "sensor_components"
 resources = {{ category = ships upkeep = {{ energy = {number(s['upkeepEnergy'])} alloys = {number(s['upkeepAlloys'])} }} }}
 min_upgrade_cost = {{ alloys = 1 }} ai_ship_data = {{ min = 0 max = 0 }}
}}''')
        self.write('common/country_limits/ship_of_size_limits/eqn_limits.txt',f'eqn_heart_limit = {{ ship_types = {{ {size} }} base = 3 max = 3 naval_cap_fraction = 0 show = {{ is_scope_valid = yes has_origin = {self.origin} }} }}')
        slots=[];counts=Counter()
        for sec in self.d['sections']:
            for slot in sec.get('slots',[]):
                if not slot.get('enabled',True):continue
                if slot['type']=='utility':counts[slot['size']]+=1
                elif slot['type']=='auxiliary':counts['A']+=1
                else:slots.append(f'component_slot = {{ name = {quote(slot["id"])} template = "eqn_horn_{slot["size"]}" locatorname = "horn_muzzle" }}')
        self.write('common/section_templates/eqn_heart.txt',f'''ship_section_template = {{
 key = "EQN_HEART_SECTION" ship_size = {size} fits_on_slot = mid
 entity = "stellaris_heart_entity" icon = "GFX_ship_part_core_mid" should_draw_components = no
 {' '.join(slots)}
 small_utility_slots = {counts['S']} medium_utility_slots = {counts['M']} large_utility_slots = {counts['L']} aux_utility_slots = {counts['A']}
 resources = {{ category = ship_sections cost = {{ alloys = 0 }} }} ai_weight = {{ weight = 100 }}
}}''')
        self.localized('EQN_HEART_SECTION','Crystal Alicorn Body')
        # Invisible native turrets: every slot follows the actual exported horn bone.
        slot_sizes={**SIZES,'W':'planet_killer'}
        self.write('common/component_slot_templates/eqn_slots.txt','\n'.join(f'eqn_horn_{k} = {{ size = {v} component = weapon {"is_fixed = yes" if k == "W" else ""} }}' for k,v in slot_sizes.items()))
    def components(self):
        output=[];sets=[]
        utility_icons={'eqn_crystal_core':'crystal_core','eqn_royal_mind':'royal_mind','eqn_crystal_wings':'crystal_wings'}
        for key,icon in utility_icons.items():
            sets.append(f'component_set = {{ key = "{key}" icon = "GFX_eqn_component_{icon}" icon_frame = 1 }}');self.localized(key,key.replace('eqn_','').replace('_',' ').title())
        for size in SIZES:
            sets.append(f'component_set = {{ key = "EQN_HORN_{size}" icon = "GFX_eqn_component_horn_{size.lower()}" icon_frame = 1 }}')
            self.localized('EQN_HORN_'+size,'Crystal Horn · '+size)
        sets.append('component_set = { key = "EQN_HORN_W" icon = "GFX_eqn_component_horn_w" icon_frame = 1 }')
        self.localized('EQN_HORN_W','Crystal Horn · W Planet Cracker')
        for i,t in enumerate(self.stages):
            req=f'prerequisites = {{ {t["id"]} }}' if t['id'] else ''
            nxt=f'upgrades_to = "EQN_CORE_{i+1}"' if i<len(self.stages)-1 else ''
            hull_add=max(0,t['hull']-self.d['ship']['hullPoints'])
            hull_regen=t.get('hullRegen',t['regen']);armor_regen=t.get('armorRegen',t['regen']);shield_regen=t.get('shieldRegen',t['regen'])
            output.append(f'''utility_component_template = {{
 key = "EQN_CORE_{i}" size = small component_set = "eqn_crystal_core" icon = "GFX_eqn_component_crystal_core" icon_frame = 1
 power = {number(self.m['ship']['power'])} {self.restriction()} {req} {nxt}
 modifier = {{ ship_hull_add = {number(hull_add)} ship_armor_add = {number(t['armor'])} ship_shield_add = {number(t['shield'])} }}
 ship_modifier = {{ ship_hull_regen_add_static = {number(hull_regen)} ship_armor_regen_add_static = {number(armor_regen)} ship_shield_regen_add_static = {number(shield_regen)} ship_armor_hardening_add = 1 ship_shield_hardening_add = 1 }}
 resources = {{ category = ship_components cost = {{ alloys = {10*(i+1)} }} }} ai_weight = {{ weight = {1000*(i+1)} }}
}}''')
            self.localized('EQN_CORE_'+str(i),t['name']+' · Crystal Core','The heart of this body. Refit after researching an upgrade; regeneration is per day.')
            for size,word in SIZES.items():
                key=f'EQN_HORN_{size}_{i}';nxt=f'upgrades_to = "EQN_HORN_{size}_{i+1}"' if i<len(self.stages)-1 else ''
                # Stellaris exposes this component property specifically for the
                # fleet-power estimator. Normalize only the estimator's weapon
                # contribution to Awakening damage so the late-stage product
                # does not overflow to a displayed power of 1. Combat damage is
                # deliberately left untouched.
                power_scale=min(1.0,self.stages[0]['weaponDamage']/t['weaponDamage'])
                power_scale_text=f'{power_scale:.9f}'.rstrip('0').rstrip('.')
                output.append(f'''weapon_component_template = {{
 key = "{key}" size = {word} type = {"point_defence" if size == "P" else "instant"} component_set = "EQN_HORN_{size}"
 icon = "GFX_eqn_component_horn_{size.lower()}" icon_frame = 1 {self.restriction()} {req} {nxt}
 damage = {{ min = {number(t['weaponDamage']*FACTORS[size])} max = {number(t['weaponDamage']*FACTORS[size])} }}
 military_power_multiplier = {power_scale_text}
 windup = {{ min = 0 max = 0 }} total_fire_time = {number(t['weaponCooldown'])}
 range = {number(t['weaponRange'])} accuracy = 1 tracking = 1 firing_arc = 360
 {'point_defence_targets = { "missile" "strike_craft" }' if size == "P" else ""}
 hull_damage = 1 armor_damage = 1 shield_damage = 1
 projectile_gfx = "{"red_laser_pd" if size == "P" else "infrared_laser_s"}" tags = {{ weapon_type_energy {"weapon_type_point_defense" if size == "P" else ""} }}
 resources = {{ category = ship_components cost = {{ alloys = {i+1} }} }} ai_weight = {{ weight = {1000*(i+1)} }}
}}''')
                self.localized(key,t['name']+' · '+size+' Horn','Omnidirectional horn emission. Raw damage is applied to every defense layer; no boss phase is bypassed.')
        output.append(f'''weapon_component_template = {{
 key = "EQN_HORN_W" size = planet_killer type = planet_killer use_ship_main_target = no component_set = "EQN_HORN_W"
 icon = "GFX_eqn_component_horn_w" icon_frame = 1 {self.restriction()}
 damage = {{ min = 0 max = 0 }}
 windup = {{ min = {number(5/self.m['ship']['planetKillerSpeedMultiplier'])} max = {number(5/self.m['ship']['planetKillerSpeedMultiplier'])} }}
 total_fire_time = 15 accuracy = 1
 planet_destruction_gfx = "eqn_shatter_planet_gfx" ai_weight = {{ weight = 10000 }}
}}
utility_component_template = {{ key = "EQN_ROYAL_MIND" size = small component_set = "eqn_royal_mind" icon = "GFX_eqn_component_royal_mind" icon_frame = 1 ship_behavior = "{self.m['ship']['behavior']}" {self.restriction()} ai_weight = {{ weight = 10000 }} }}
utility_component_template = {{ key = "EQN_CRYSTAL_WINGS" size = small component_set = "eqn_crystal_wings" icon = "GFX_eqn_component_crystal_wings" icon_frame = 1 {self.restriction()} ai_weight = {{ weight = 10000 }} }}''')
        # The editor value controls charge speed, not the whole destruction
        # sequence.  Keep the vanilla World Cracker's visible firing phases so
        # the shot cannot collapse into a near-instant effect.  The explicit
        # component windup/total_fire_time above follows the installed
        # Cultivation mod precedent; omitting them made a non-colossus hull
        # complete its planet-killer order immediately.
        speed=self.m['ship']['planetKillerSpeedMultiplier']
        self.write('gfx/projectiles/planet_destruction/eqn_planet_destruction.txt',f'''eqn_shatter_planet_gfx = {{
 texture = "gfx/models/combat_items/shatter_planet_laser.dds"
 color = {{ 1.0 1.0 1.0 1.0 }} planet_dissolve_color_mult = {{ 1.5 0.75 0.4 }}
 windup_entity = "colossus_shatter_planet_windup_entity" ship_fire_entity = "colossus_shatter_planet_muzzle_entity"
 planet_hit_entity = "colossus_shatter_planet_hit_entity" megastructure_hit_entity = "colossus_shatter_planet_hit_entity"
 windup = {{ duration = {number(5/speed)} }}
 start = {{ duration = 2 width = {{ 0.0 1.0 1.0 12.0 }} texture_scroll_speed = {{ 0.0 1.5 }} texture_tiling = 2.0 alpha = {{ 0.0 0.0 0.25 4.0 1.0 2.0 }} }}
 in_progress = {{ duration = 10 width = {{ 0.0 12.0 0.5 9.0 1.0 12.0 }} texture_scroll_speed = {{ 0.0 1.5 }} texture_tiling = {{ 0.0 2.0 }} alpha = {{ 0.0 2.0 0.5 2.2 1.0 2.0 }} }}
 end = {{ duration = 2 width = {{ 0.0 11.0 1.0 0.0 }} texture_scroll_speed = 1.5 texture_tiling = 2.0 alpha = {{ 0.0 2.1 0.05 10.0 0.7 1.0 1.0 0.0 }} }}
 fade = {{ fade_in = {{ 0.0 50.0 }} fade_out = {{ 1.0 50.0 }} }}
}}''')
        self.write('common/scripted_triggers/eqn_planet_killer.txt','''can_destroy_planet_with_EQN_HORN_W = {
 can_destroy_planet_with_PLANET_KILLER_CRACKER = yes
}''')
        self.write('common/on_actions/eqn_planet_killer.txt','''on_destroy_planet_with_EQN_HORN_W = {
 events = {
  toxoids.8016
  crisis.5015
  planet_destruction.110
  origin.3245
  planet_destruction.600
  planet_destruction.100
  awareness.150
  timeline.27
 }
}
on_destroy_planet_with_EQN_HORN_W_queued = { events = { fircon.5035 } }
on_destroy_planet_with_EQN_HORN_W_unqueued = { }''')
        self.localized('EQN_HORN_W_ACTION','Crystal Shatter World')
        self.localized('FLEETORDER_DESTROY_PLANET_WITH_EQN_HORN_W',"Preparing the Heart's crystal shatter on $PLANET|Y$")
        self.localized('EQN_ROYAL_MIND','Royal Battle Mind');self.localized('EQN_CRYSTAL_WINGS','Crystal Wings')
        self.write('common/component_templates/eqn_components.txt','\n'.join(output));self.write('common/component_sets/eqn_components.txt','\n'.join(sets))
    def technology(self):
        output=[]
        for t in self.m['technologies']:
            req=[t['prerequisite']] if t['prerequisite'] else []
            external=t['externalPrerequisite']
            if external and (not external.startswith('tech_dark_matter_') or self.m['compatibility']['mode']=='acot'):req.append(external)
            output.append(f'''{t['id']} = {{
 area = {t['area']} category = {{ {dict(physics='particles',society='biology',engineering='voidcraft')[t['area']]} }} tier = {t['tier']} cost = {number(t['cost'])} weight = {number(t['weight'])}
 prerequisites = {{ {' '.join(req)} }} potential = {{ is_ai = no has_origin = {self.origin} }}
 ai_weight = {{ weight = 0 }} is_rare = yes icon = "{t['icon']}"
 weight_modifier = {{ factor = 1 }}
}}''')
            self.localized(t['id'],t['name'],t['description'])
        for bt in self.m['bodyTechnologies']:
            body,req,cost=bt['body'],bt['prerequisite'],bt['cost']
            output.append(f'tech_eqn_body_{body} = {{ area = society category = {{ biology }} tier = {bt['tier']} cost = {cost} weight = {bt['weight']} prerequisites = {{ {req} }} potential = {{ is_ai = no has_origin = {self.origin} }} ai_weight = {{ weight = 0 }} icon = "{bt['icon']}" }}')
            self.localized('tech_eqn_body_'+str(body),bt['name'],f'Unlock body {body} through a planetary decision. Awakening happens at the capital; an existing body can switch form at any owned colony in its system.')
        self.write('common/technology/eqn_technologies.txt','\n'.join(output))
    def leaders(self):
        output=[]
        ruler=self.m['leader']['rulerBonuses']
        # Native leader traits use one triggered army block, scaled by a script value.
        # Select the highest owned stage even when an edited tree skips a lower tier.
        stage_values=[]
        for i,stage in enumerate(self.m['technologies']):
            later=' '.join('has_technology = '+t['id'] for t in self.m['technologies'][i+1:])
            exclusion='NOR = { '+later+' }' if later else ''
            stage_values.append('modifier = { exists = owner owner = { has_technology = '+stage['id']+' '+exclusion+' } add = '+number(stage['groundMultiplier']-1)+' }')
        self.write('common/script_values/eqn_ground.txt','eqn_ground_progression = { base = 0\n'+'\n'.join(stage_values)+'\n}')
        for ti,t in enumerate(self.m['leader']['traits']):
            trait_icon={'crystal_pony':'GFX_eqn_heart','crystal_research':'GFX_eqn_trait_research'}[t.get('iconStyle','crystal_pony')]
            ground_bonuses=[]
            ruler_bonuses=''
            if ti == 0:
                ground_bonuses.append('triggered_army_modifier = { potential = { leader_class = commander } army_damage_mult = 1 army_health = 1 army_morale = 1 mult = value:eqn_ground_progression }')
                ruler_bonuses=f'''triggered_councilor_modifier = {{
 potential = {{ is_ruler = yes }}
 all_technology_research_speed = {number(ruler['researchSpeed'])}
 country_unity_produces_mult = {number(ruler['unity'])}
 country_naval_cap_mult = {number(ruler['navalCapacity'])}
 starbase_shipyard_build_speed_mult = {number(ruler['shipBuildSpeed'])}
}}'''
            output.append(f'''{t['id']} = {{
 icon = {{ layer = {{ icon = {trait_icon} }} }} leader_class = {{ scientist official commander }}
 randomized = no selectable_weight = {{ weight = 0 }} ai_weight = {{ weight = 0 }}
 immortal_leaders = {'yes' if t['immortal'] else 'no'}
 fleet_modifier = {{ ship_weapon_damage = {number(t['weaponDamage'])} ship_fire_rate_mult = {number(t['fireRate'])} ship_weapon_range_mult = {number(t['weaponRange'])} }}
 army_modifier = {{ army_damage_mult = {number(t['armyDamage'])} army_health = {number(t['armyHealth'])} army_morale = {number(t.get('armyMorale',0))} }}
 {ruler_bonuses}
 {' '.join(ground_bonuses)}
}}''')
            self.localized(t['id'],t['name'],t['description'])
        self.write('common/traits/eqn_leaders.txt','\n'.join(output))
    def mechanics(self):
        # Filled from the installed Gray form-switch contract: one named flag per body.
        effects=[];decisions=[];armies=[];modifiers=[]
        ground=self.m['ground']
        for body in range(1,4):
            flag=f'eqn_body_{body}';leaderflag=f'eqn_commander_{body}'
            self.localized(flag,self.d['ship']['name']+f' · Body {body}')
            traits=' '.join('trait = '+t['id'] for t in self.m['leader']['traits'])
            commander_key=self.localized(f'eqn_commander_name_{body}',self.m['leader']['name']+' · '+str(body))
            effects.append(f'''eqn_ensure_commander_{body} = {{
 if = {{ limit = {{ NOT = {{ any_owned_leader = {{ has_leader_flag = {leaderflag} }} }} }}
 create_leader = {{ class = commander name = {commander_key} species = owner_main_species gender = female skill = {self.m['origin']['startingLeaderLevel']} traits = {{ {traits} }}
 effect = {{ set_leader_flag = {leaderflag} set_age = 25 change_leader_portrait = eqn_heart_portrait }} }} }}
 random_owned_leader = {{ limit = {{ has_leader_flag = {leaderflag} }} save_event_target_as = eqn_active_commander }}
}}''')
            # ROOT can be either starting country or decision planet; save explicit country target.
            effects.append(f'''eqn_create_space_{body} = {{
 save_event_target_as = eqn_body_owner
 eqn_ensure_commander_{body} = yes
 if = {{ limit = {{ NOT = {{ exists = event_target:eqn_body_location }} }} capital_scope = {{ save_event_target_as = eqn_body_location }} }}
 create_fleet = {{ name = {flag} settings = {{ can_upgrade = yes can_change_composition = no can_change_leader = yes spawn_debris = no }}
 effect = {{ set_owner = event_target:eqn_body_owner set_fleet_flag = {flag}
 create_ship = {{ name = {flag} random_existing_design = {self.size} prefix = no upgradable = yes effect = {{ set_ship_flag = {flag} }} }}
 assign_leader = event_target:eqn_active_commander
 set_location = event_target:eqn_body_location
 }} }}
}}''')
            armies.append(f'''eqn_ground_{body} = {{
 damage = {number(ground['damage'])} health = {number(ground['health'])} morale = {number(ground['morale'])} morale_damage = {number(ground['moraleDamage'])}
 collateral_damage = {number(ground['collateralDamage'])} war_exhaustion = {number(ground['warExhaustion'])}
 has_species = no time = 1 icon = GFX_eqn_ground
 resources = {{ category = armies upkeep = {{ energy = {number(ground['upkeepEnergy'])} }} }}
 potential_country = {{ always = no }}
}}''')
            self.localized(f'eqn_ground_{body}',ground['name']+f' {body}', 'A transportable assault incarnation of the princess. Use a decision on any owned colony in its system to switch forms, then embark and land it on a hostile planet like a normal assault army.')
            req=f'has_technology = tech_eqn_body_{body}' if body>1 else 'always = yes'
            own=f'owner = {{ is_ai = no has_origin = {self.origin} {req} }}'
            none=f'owner = {{ NOR = {{ any_owned_ship = {{ has_ship_flag = {flag} }} any_owned_army = {{ army_type = eqn_ground_{body} }} }} }}'
            have_ship=f'owner = {{ any_owned_fleet = {{ has_fleet_flag = {flag} num_ships = 1 is_in_combat = no NOT = {{ has_hp_percentage < 1.0 }} solar_system = {{ is_same_value = root.solar_system }} }} }}'
            have_army=f'owner = {{ any_owned_army = {{ army_type = eqn_ground_{body} exists = planet planet = {{ is_same_value = root has_ground_combat = no }} }} }}'
            create=f'owner = {{ eqn_create_space_{body} = yes }}'
            ground_effect=f'''save_event_target_as = eqn_body_location
 owner = {{ save_event_target_as = eqn_body_owner eqn_ensure_commander_{body} = yes
 every_owned_fleet = {{ limit = {{ has_fleet_flag = {flag} }} if = {{ limit = {{ exists = leader }} leader = {{ unassign_leader = this }} }} delete_fleet = this }} }}
 create_army = {{ name = {flag} owner = event_target:eqn_body_owner type = eqn_ground_{body}
 effect = {{ set_army_flag = {flag} assign_leader = event_target:eqn_active_commander }} }}'''
            space_effect=f'save_event_target_as = eqn_body_location owner = {{ every_owned_army = {{ limit = {{ army_type = eqn_ground_{body} }} if = {{ limit = {{ exists = leader }} leader = {{ unassign_leader = this }} }} remove_army = yes }} eqn_create_space_{body} = yes }}'
            for action,allow,effect,label,desc,cost in [
                ('awaken',none,create,f'Awaken Body {body}',f'Create incarnation {body} at the capital. It must not already exist as a ship or army.',self.m['ship']['cloneAlloys']),
                ('ground',have_ship,ground_effect,f'Body {body} to Ground', 'On any owned colony, transform the healthy hero ship in this system into a transportable assault army. Embark it, then right-click a hostile planet to land. The same commander continues.',0),
                ('space',have_army,space_effect,f'Body {body} to Space','Transform the ground body stationed on this owned colony into its hero ship. Embarked armies must land on an owned colony first.',0),
            ]:
                key=f'eqn_deploy_ground_{body}' if action=='ground' else f'eqn_{action}_{body}';self.localized(key,label,desc)
                location='is_capital = yes' if action=='awaken' else 'is_colony = yes'
                decisions.append(f'''{key} = {{
 icon = "decision_robot_assembly_control" enactment_time = 0
 resources = {{ category = decisions cost = {{ alloys = {number(cost)} }} }}
 potential = {{ {location} {own} }}
 allow = {{ {allow} }} effect = {{ custom_tooltip = {key}_desc hidden_effect = {{ {effect} }} }} ai_weight = {{ weight = 0 }}
}}''')
        self.write('common/scripted_effects/eqn_mechanics.txt','\n'.join(effects))
        self.write('common/decisions/eqn_bodies.txt','\n'.join(decisions))
        self.write('common/armies/eqn_ground.txt','\n'.join(armies))
        # A ruler can reject additional ordinary traits once its normal trait
        # selections are full. Keep the permanent identity/country/fleet/army
        # bonuses in the primary Heart trait; commander-only progression traits
        # are assigned when the separate physical-body leader is created.
        ruler_trait=self.m['leader']['traits'][0]['id']
        ruler_traits=f'''if = {{ limit = {{ NOT = {{ has_trait = {ruler_trait} }} }} add_trait = {{ trait = {ruler_trait} show_message = no }} }}'''
        ruler_name=self.localized('eqn_ruler_name',self.m['leader']['name'])
        self.write('common/on_actions/eqn_actions.txt','''on_game_start_country = { events = { eqn_go.1 eqn_go.2 } }
on_ruler_set = { events = { eqn_go.2 } }
on_single_player_save_game_load = { events = { eqn_go.4 } }''')
        self.write('events/eqn_mechanical_events.txt',f'''namespace = eqn_go
# Hidden implementation hooks only. No optional story events.
country_event = {{ id = eqn_go.1 hide_window = yes is_triggered_only = yes
 trigger = {{ is_ai = no has_origin = {self.origin} NOT = {{ has_country_flag = eqn_initialized }} }}
 immediate = {{ set_country_flag = eqn_initialized country_event = {{ id = eqn_go.3 days = 2 }} }}
}}
country_event = {{ id = eqn_go.3 hide_window = yes is_triggered_only = yes
 trigger = {{ is_ai = no has_origin = {self.origin} }}
 immediate = {{ if = {{ limit = {{ NOR = {{ any_owned_ship = {{ has_ship_flag = eqn_body_1 }} any_owned_army = {{ army_type = eqn_ground_1 }} }} }} eqn_create_space_1 = yes }} }}
}}
country_event = {{ id = eqn_go.2 hide_window = yes is_triggered_only = yes
 trigger = {{ is_ai = no has_origin = {self.origin} exists = ruler }}
 immediate = {{ ruler = {{ set_name = {ruler_name} change_leader_portrait = eqn_heart_portrait set_age = 25 set_leader_flag = eqn_heart_ruler {ruler_traits} }} }}
}}
event = {{ id = eqn_go.4 hide_window = yes is_triggered_only = yes
 immediate = {{ every_country = {{ limit = {{ is_ai = no has_origin = {self.origin} }} country_event = {{ id = eqn_go.2 }} }} }}
}}
''')
    def graphics(self):
        art=self.m['artwork'];icons={i['id']:i for i in art['icons']}
        sprites=[]
        for style,name,usage in [('crystal_pony','eqn_heart','trait'),('crystal_research','eqn_research','component'),('crystal_research','eqn_trait_research','trait'),('three_incarnations','eqn_incarnations','trait')]:
            dst=f'gfx/interface/icons/{name}.dds';target=self.out/dst;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(icon_texture(icons[style],usage),target)
            sprites.append(f'spriteType = {{ name = "GFX_{name}" texturefile = "{dst}" }}')
        portrait=self.out/'gfx/portraits/eqn_heart.dds';portrait.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(asset(self.m['leader']['portrait']),portrait)
        shutil.copy2(icon_texture(icons['crystal_pony'],'origin'),self.out/'gfx/interface/icons/eqn_origin.dds')
        ground_icon=next((i for i in icons.values() if i['texture']==self.m['ground']['icon']),None)
        if ground_icon is None:raise ValueError('Select a registered ground icon with native-size derivatives')
        ground_path='gfx/interface/icons/eqn_ground.dds';shutil.copy2(icon_texture(ground_icon,'army'),self.out/ground_path)
        sprites.append(f'spriteType = {{ name = "GFX_eqn_ground" texturefile = "{ground_path}" }}')
        component_dir=self.out/'gfx/interface/icons/ship_parts';component_dir.mkdir(parents=True,exist_ok=True)
        for icon in self.component_icons.values():
            name='eqn_component_'+icon['id'];dst=component_dir/(name+'.dds');shutil.copy2(fixed_texture(icon['texture'],58,'component'),dst)
            sprites.append(f'spriteType = {{ name = "GFX_{name}" texturefile = "gfx/interface/icons/ship_parts/{name}.dds" }}')
        technology_dir=self.out/'gfx/interface/icons/technologies';technology_dir.mkdir(parents=True,exist_ok=True)
        for icon in self.technology_icons.values():
            shutil.copy2(fixed_texture(icon['texture'],52,'technology'),technology_dir/(icon['id']+'.dds'))
        p=self.out/'gfx/event_pictures/eqn_origin.dds';p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(asset(art['originPicture']),p)
        sprites.append('spriteType = { name = "GFX_eqn_origin" texturefile = "gfx/event_pictures/eqn_origin.dds" alwaystransparent = yes }')
        self.write('interface/eqn_icons.gfx','spriteTypes = { '+' '.join(sprites)+' }')
        self.write('eqn_art_provenance.json',asset(art['provenance']).read_text())
        rebuilt_native=ROOT/'outputs/heart_animation_editor_rebuild/native'
        native=rebuilt_native if (rebuilt_native/'native_validation.json').exists() else ROOT/'outputs/heart/native'
        if not (native/'stellaris_heart.mesh').exists():
            # TK2 build uses the previously declared, hydrated native ZIP.
            archive=ROOT/'outputs/heart/heart_native.zip'
            if not archive.exists():raise ValueError('Hydrate the native Heart bundle before export')
            with zipfile.ZipFile(archive) as z:
                native=self.out.parent/'native_source';native.mkdir(exist_ok=True)
                for item in z.infolist():
                    if Path(item.filename).suffix in ('.mesh','.anim','.dds','.gfx','.asset') or Path(item.filename).name == 'native_validation.json':
                        dest=native/Path(item.filename).name;dest.write_bytes(z.read(item))
        validation=native/'native_validation.json'
        if not validation.exists():raise ValueError('Native Heart validation is missing; rebuild/export the 50-joint asset')
        evidence=json.loads(validation.read_text())
        if evidence.get('nativeJointBudget',{}).get('joints',100)>50:raise ValueError('Stellaris accepts at most 50 joints; rebuild native Heart')
        mesh_entry=next(item for item in evidence['files'] if item['name']=='stellaris_heart.mesh')
        if hashlib.sha256((native/'stellaris_heart.mesh').read_bytes()).hexdigest()!=mesh_entry['sha256']:raise ValueError('Native mesh differs from its validated export')
        self.write('eqn_native_validation.json',json.dumps(evidence,indent=2))
        dest=self.out/'gfx/models/ships/stellaris_heart';dest.mkdir(parents=True,exist_ok=True)
        for p in native.iterdir():
            if p.is_file() and p.suffix in ('.mesh','.anim','.dds','.gfx','.asset'):shutil.copy2(p,dest/p.name)
        self.write('gfx/models/ships/stellaris_heart/eqn_frame.asset',f'entity = {{ name = "eqn_heart_frame_entity" locator = {{ name = "part1" position = {{ 0 0 0 }} }} scale = {number(self.d["ship"]["entityScale"])} }}')
        # Native animation state bindings follow the GUI's approved game mapping.
        p=dest/'stellaris_heart.asset';text=p.read_text()
        text=re.sub(r'(file = ")gfx/models/ships/stellaris_heart/', r'\1', text)
        for state,clip in self.d['animation']['gameStateMap'].items():
            if clip not in {x['id'] for x in self.d['animation']['clips']}:raise ValueError('Unknown native animation clip')
            text=re.sub(r'(name = "'+re.escape(state)+r'" animation = ")[^"]+(" )',lambda match:match[1]+clip+match[2],text)
        p.write_text(text)

def parse_check(path):
    text=path.read_text(encoding='utf-8-sig');stack=[];quoted=False;escape=False
    for line_no,line in enumerate(text.splitlines(),1):
        for c in line:
            if escape:escape=False;continue
            if quoted and c=='\\':escape=True;continue
            if c=='"':quoted=not quoted;continue
            if not quoted:
                if c=='#':break
                if c=='{':stack.append(line_no)
                if c=='}':
                    if not stack:raise ValueError(f'{path}:{line_no}: unexpected closing brace')
                    stack.pop()
    if quoted or stack:raise ValueError(f'{path}: unterminated string/block {stack}')

def validate_export(out,d,game=None):
    checked=0
    for p in out.rglob('*'):
        if p.is_file() and p.suffix in ('.txt','.asset','.gfx','.mod'):
            parse_check(p);checked+=1
    missing=[]
    for p in out.rglob('*'):
        if p.is_file() and p.suffix in ('.txt','.asset','.gfx'):
            for ref in re.findall(r'(?:texturefile|file)\s*=\s*"([^"]+)"',p.read_text(encoding='utf-8-sig')):
                if ref.startswith('gfx/') and not (out/ref).is_file() and not (game and (game/ref).is_file()):missing.append(ref)
                elif p.suffix=='.asset' and ref.endswith('.anim') and not (p.parent/ref).is_file():missing.append(ref)
    if missing:raise ValueError('Unresolved asset references: '+', '.join(sorted(set(missing))))
    if game:
        roots=[game]
        if d['mod']['compatibility']['mode']=='acot':
            workshop=game.parents[1]/'workshop/content/281990'
            roots += [workshop/x for x in ('1419304439','1481972266','1504307690')]
            for r in roots[1:]:
                if not (r/'descriptor.mod').exists():raise ValueError(f'Required installed ACOT mod missing: {r}')
        techs=set()
        for r in roots:
            for f in (r/'common/technology').glob('*.txt'):
                techs.update(re.findall(r'^\s*(tech_\w+)\s*=\s*{',f.read_text(encoding='utf-8-sig',errors='replace'),re.M))
        missing=[t['externalPrerequisite'] for t in d['mod']['technologies'] if t['externalPrerequisite'] and t['externalPrerequisite'] not in techs and (d['mod']['compatibility']['mode']=='acot' or not t['externalPrerequisite'].startswith('tech_dark_matter_'))]
        if missing:raise ValueError('Unresolved installed technology prerequisites: '+', '.join(missing))
    files=sorted(p for p in out.rglob('*') if p.is_file())
    package_hash=hashlib.sha256()
    for p in files:
        package_hash.update(p.relative_to(out).as_posix().encode()+b'\0'+hashlib.sha256(p.read_bytes()).digest())
    stages=[{'name':'Awakening','weaponDamage':d['mod']['ship']['weaponDamage']},*({'name':t['name'],'weaponDamage':t['weaponDamage']} for t in d['mod']['technologies'])]
    base_damage=stages[0]['weaponDamage']
    power_stages=[{
        **stage,
        'militaryPowerMultiplier':min(1.0,base_damage/stage['weaponDamage']),
        'estimatorWeaponDamage':stage['weaponDamage']*min(1.0,base_damage/stage['weaponDamage']),
    } for stage in stages]
    return {'staticValidation':'passed','packageSha256':package_hash.hexdigest(),'scriptFilesChecked':checked,'files':len(files),'bytes':sum(p.stat().st_size for p in files),'configSha256':hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest(),'species':[{'name':s['name'],'portraits':sum(p['enabled'] for p in s['portraits'])} for s in d['mod']['species']],'technologyCount':len(d['mod']['technologies'])+2,'weaponSlots':sum(s.get('enabled',True) and s['type'] not in ('utility','auxiliary') for sec in d['sections'] for s in sec.get('slots',[])),'bodyCap':3,'groundSharesBodyCap':True,'narrativeEventsAdded':False,'gameLoadVerified':False,'soloBossVictoryVerified':False,'compatibilityMode':d['mod']['compatibility']['mode'],'militaryPowerEstimator':{'strategy':'Native per-weapon military_power_multiplier normalizes estimator-only damage to Awakening while real damage is unchanged.','combatStatsChanged':False,'stages':power_stages},'notes':['Static checks do not prove game load or victory.','SBTG installed descriptor targets 4.1; verify the actual mod combination in 4.4.6.']}

def build(config=CONFIG,output=None,game=None):
    d=validate_config(json.loads(Path(config).read_text()));out=Path(output or ROOT/'outputs/eqn_go');out.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='eqn-build-',dir=out.parent))
    try:
        Compiler(d,staging).build();report=validate_export(staging,d,game)
        if out.exists():
            backup=out.with_name(out.name+'.previous')
            if backup.exists():shutil.rmtree(backup)
            out.rename(backup)
        staging.rename(out)
        archive=out.parent/'eqn_go.zip'
        archive_stage=archive.with_suffix('.zip.building')
        with zipfile.ZipFile(archive_stage,'w',zipfile.ZIP_DEFLATED) as z:
            for p in out.rglob('*'):
                if p.is_file():z.write(p,'eqn_go/'+str(p.relative_to(out)))
            z.writestr('eqn_go.mod',(out/'descriptor.mod').read_text()+'path = "mod/eqn_go"\n')
        archive_stage.replace(archive)
        report['archive']=str(archive)
        report_path=out.parent/'eqn_build_report.json';report_path.write_text(json.dumps(report,indent=2)+'\n')
        (out.parent/'eqn_go.mod').write_text((out/'descriptor.mod').read_text()+f'path = {quote(str(out.resolve()))}\n')
        return report
    finally:
        if staging.exists():shutil.rmtree(staging)

def install(out):
    if sys.platform!='linux':raise ValueError('Install is intended for TK2 Linux; use Build on Mac or Deploy on TK2')
    # Preserve previous release; do not alter the user's active playset or third-party mods.
    user=Path.home()/'.local/share/Paradox Interactive/Stellaris'
    if not user.exists():user=Path.home()/'.local/share/Paradox Interactive/Stellaris'
    mods=user/'mod';mods.mkdir(parents=True,exist_ok=True);dest=mods/'eqn_go'
    stage=mods/'eqn_go.installing'
    if stage.exists():shutil.rmtree(stage)
    shutil.copytree(out,stage)
    if dest.exists():
        backup=mods/'eqn_go.previous'
        if backup.exists():shutil.rmtree(backup)
        dest.rename(backup)
    stage.rename(dest)
    descriptor=mods/'eqn_go.mod';descriptor.write_text((dest/'descriptor.mod').read_text()+f'path = {quote(str(dest))}\n')
    launcher_thumbnail={'databaseFound':False,'rowsUpdated':0}
    launcher_db=user/'launcher-v2.sqlite'
    if launcher_db.exists():
        launcher_thumbnail['databaseFound']=True
        launcher_backup=user/'launcher-v2.pre-eqn-local-thumbnail.sqlite'
        if not launcher_backup.exists():shutil.copy2(launcher_db,launcher_backup)
        with sqlite3.connect(launcher_db,timeout=5) as database:
            updated=database.execute(
                '''UPDATE mods SET thumbnailPath = ?
                   WHERE source = 'local' AND (gameRegistryId = 'mod/eqn_go.mod' OR dirPath = ?)''',
                (str(dest/'thumbnail.png'),str(dest)),
            )
        launcher_thumbnail.update({'rowsUpdated':updated.rowcount,'backup':str(launcher_backup),'path':str(dest/'thumbnail.png')})
    return {'installed':str(dest),'descriptor':str(descriptor),'activePlaysetChanged':False,'launcherThumbnail':launcher_thumbnail}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=CONFIG);parser.add_argument('--output',type=Path);parser.add_argument('--check-config',action='store_true');parser.add_argument('--install',action='store_true');parser.add_argument('--game',type=Path)
    args=parser.parse_args()
    try:
        if args.check_config:
            validate_config(json.loads(args.config.read_text()));print(json.dumps({'valid':True}));return
        game=args.game
        if not game and sys.platform=='linux':game=Path.home()/'.local/share/Steam/steamapps/common/Stellaris'
        report=build(args.config,args.output,game)
        if args.install:
            report['deployment']=install(args.output or ROOT/'outputs/eqn_go')
            report_path=(args.output or ROOT/'outputs/eqn_go').parent/'eqn_build_report.json'
            report_path.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
    except (ValueError,OSError,KeyError,TypeError) as e:
        print(str(e),file=sys.stderr);raise SystemExit(1)
if __name__=='__main__':main()

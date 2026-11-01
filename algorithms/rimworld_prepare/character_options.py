"""Read official installed RimWorld definitions; also runs over SSH via stdin."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET

TYPES = {'TraitDef': 'traits', 'BackstoryDef': 'backstories', 'GeneDef': 'genes',
         'XenotypeDef': 'xenotypes', 'BeardDef': 'beards', 'TattooDef': 'tattoos',
         'ColorDef': 'favoriteColors'}
TEMPLATE_TYPES = {'GeneTemplateDef', 'SkillDef', 'ChemicalDef'}


def collect(data):
    nodes, parents = [], {}
    expansions = []
    for pack in sorted(data.iterdir()):
        if not (pack / 'Defs').is_dir():
            continue
        expansions.append(pack.name)
        for path in sorted((pack / 'Defs').rglob('*.xml')):
            try:
                root = ET.parse(path).getroot()
            except ET.ParseError:
                continue
            for node in root:
                if node.tag not in TYPES and node.tag not in TEMPLATE_TYPES:
                    continue
                if node.get('Name'):
                    parents[node.get('Name')] = node
                nodes.append((node, pack.name))

    def inherited(node, field, seen=()):
        value = node.find(field)
        if value is not None:
            return value
        parent = node.get('ParentName')
        if parent in parents and parent not in seen:
            return inherited(parents[parent], field, (*seen, parent))
        return None

    def text(node, field, default=''):
        value = inherited(node, field)
        return value.text or default if value is not None else default

    def items(node, field):
        value = inherited(node, field)
        return [n.text for n in value if n.text] if value is not None else []

    def gene_fields(node):
        return dict(category=text(node, 'displayCategory', 'Other'),
                    complexity=int(text(node, 'biostatCpx', '0')),
                    metabolism=int(text(node, 'biostatMet', '0')),
                    archites=int(text(node, 'biostatArc', '0')),
                    exclusionTags=items(node, 'exclusionTags'),
                    prerequisite=text(node, 'prerequisite'))

    result = {key: [] for key in TYPES.values()}
    for node, pack in nodes:
        name = node.findtext('defName')
        if not name or node.get('Abstract', '').lower() == 'true':
            continue
        if node.tag not in TYPES:
            continue
        entry = {'defName': name, 'label': text(node, 'label', name), 'source': pack,
                 'description': text(node, 'description')}
        if node.tag == 'TraitDef':
            degrees = inherited(node, 'degreeDatas')
            entry['degrees'] = [{'degree': int(n.findtext('degree', '0')), 'label': n.findtext('label', name),
                                 'description': n.findtext('description', '')} for n in degrees] if degrees is not None else []
            entry['conflicts'] = items(node, 'conflictingTraits')
        elif node.tag == 'BackstoryDef':
            entry.update(label=text(node, 'title', name), slot=text(node, 'slot', 'Childhood'),
                         workDisables=text(node, 'workDisables'))
        elif node.tag == 'GeneDef':
            entry.update(gene_fields(node))
        elif node.tag == 'XenotypeDef':
            entry.update(genes=items(node, 'genes'), inheritable=text(node, 'inheritable', 'false') == 'true')
        elif node.tag == 'TattooDef':
            entry['tattooType'] = text(node, 'tattooType')
        elif node.tag == 'ColorDef':
            # Prepare Carefully's favorite-color picker uses these two types.
            if text(node, 'colorType', 'Misc') not in {'Ideo', 'Misc'}:
                continue
            entry['color'] = text(node, 'color')
        result[TYPES[node.tag]].append(entry)

    # Verified against TK2 RimWorld 1.6's GeneDefGenerator.GetFromTemplate and
    # ImpliedGeneDefs: defName and exclusion tag join the template prefix and
    # target defName with '_'; chemicals opt out via generateAddictionGenes.
    for template, pack in nodes:
        template_name = template.findtext('defName')
        if (template.tag != 'GeneTemplateDef' or not template_name
                or template.get('Abstract', '').lower() == 'true'):
            continue
        kind = text(template, 'geneTemplateType')
        if kind not in {'Skill', 'Chemical'}:
            continue
        for target, _ in nodes:
            target_name = target.findtext('defName')
            if (target.tag != kind + 'Def' or not target_name
                    or target.get('Abstract', '').lower() == 'true'):
                continue
            if kind == 'Chemical' and text(target, 'generateAddictionGenes', 'true') == 'false':
                continue
            label = text(target, 'label', text(target, 'skillLabel', target_name))
            entry = {'defName': template_name + '_' + target_name,
                     'label': text(template, 'label', template_name).replace('{0}', label),
                     'description': text(template, 'description').replace('{0}', label),
                     'source': pack, 'template': template_name, **gene_fields(template)}
            prefix = text(template, 'exclusionTagPrefix')
            if prefix:
                entry['exclusionTags'] = [prefix + '_' + target_name]
            if kind == 'Chemical':
                overrides = inherited(template, 'chemicalBiostatOverrides')
                for override in overrides if overrides is not None else []:
                    if override.findtext('chemical') == target_name:
                        for key, field in [('complexity', 'biostatCpx'), ('metabolism', 'biostatMet'), ('archites', 'biostatArc')]:
                            value = override.findtext(field)
                            if value is not None:
                                entry[key] = int(value)
            result['genes'].append(entry)
    for entries in result.values():
        entries.sort(key=lambda item: (item['label'].lower(), item['defName']))
    result['installedExpansions'] = expansions
    gene_names = {entry['defName'] for entry in result['genes']}
    result['unresolvedXenotypeGenes'] = sorted({gene for x in result['xenotypes'] for gene in x['genes']} - gene_names)
    result['scope'] = 'Installed official definitions and skill/chemical gene templates on TK2; DLC must be enabled in game. Workshop definitions and patches are not included.'
    return result


if __name__ == '__main__':
    print(json.dumps(collect(Path('/home/wishai/.local/share/Steam/steamapps/common/RimWorld/Data')), ensure_ascii=False))

"""Reviewed sculpt landmarks + crease connectivity; no bones, UVs or colors.

Coincident render vertices are welded only in the analysis graph. Garment
subsets keep the original corner data. Small residual faces are reached along
the surface, with a strong cost for crossing a sculpted crease.
"""
from pathlib import Path
import heapq
import json
import math
import runpy
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/landau_v10'


def graph(mesh):
    v = np.array([p.co[:] for p in mesh.vertices])
    f = np.array([p.vertices[:] for p in mesh.polygons])
    _, inverse = np.unique(np.round(v, 6), axis=0, return_inverse=True)
    wf = inverse[f]
    c = v[f].mean(1)
    n = np.cross(v[f[:, 1]]-v[f[:, 0]], v[f[:, 2]]-v[f[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1)[:, None], 1e-12)
    edges, links = {}, []
    for i, ids in enumerate(wf):
        for a, b in zip(ids, np.roll(ids, -1)):
            edge = tuple(sorted((int(a), int(b))))
            if edge in edges:
                j = edges[edge]
                angle = math.acos(float(np.clip(n[i] @ n[j], -1, 1)))
                links.append((j, i, angle, edge))
            else:
                edges[edge] = i
    parent = list(range(len(f)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b, angle, _ in links:
        if angle < math.radians(15):
            parent[find(a)] = find(b)
    components = np.array([find(i) for i in range(len(f))])
    return v, f, c, inverse, links, components


def audit(obj):
    v, f, c, inv, links, roots = graph(obj.data)
    rows = []
    for r, count in zip(*np.unique(roots, return_counts=True)):
        ids = np.flatnonzero(roots == r)
        p = c[ids]
        if count > 30 and p[:, 2].mean() < .58:
            rows.append(dict(id=int(r), faces=int(count), center=p.mean(0).tolist(),
                             bounds=[p.min(0).tolist(), p.max(0).tolist()]))
    print(json.dumps(sorted(rows, key=lambda x: -x['faces'])))
    return rows


# Components reviewed in front/side/back clay on the immutable 50k-face sculpt.
# IDs are graph representatives, not material IDs or bone groups. Large smooth
# panels provide unambiguous seeds; folds/trim are assigned by surface paths.
REVIEWED = {
    'Vest': [45646, 36509, 41266, 32512, 44351, 38601, 41206, 38378,
             42377, 40749, 36305, 36265, 36317, 42368],
    'Sleeve_L': [13462], 'Sleeve_R': [45499],
    'Cuff_L': [13463], 'Cuff_R': [40227],
    'Trousers': [46135],
    'Boot_L': [9947, 9360, 4611, 9971, 9099, 9147],
    'Boot_R': [9972, 8728, 10000, 6810],
    'Exclude': [41573, 41706, 46909],
}


def segment(obj):
    v, f, c, inv, links, components = graph(obj.data)
    labels = [None]*len(f)
    distance = np.full(len(f), np.inf)
    queue = []
    for name, roots in REVIEWED.items():
        for i in np.flatnonzero(np.isin(components, roots)):
            labels[i] = name
            distance[i] = 0
            heapq.heappush(queue, (0., int(i), name))
    for i in np.flatnonzero(c[:, 2] > .578):
        labels[i] = 'Exclude'; distance[i] = 0
        heapq.heappush(queue, (0., int(i), 'Exclude'))
    adjacency = [[] for _ in f]
    for a, b, angle, edge in links:
        cost = np.linalg.norm(c[a]-c[b])*(1+(angle/.20)**4)
        adjacency[a].append((b, cost)); adjacency[b].append((a, cost))
    while queue:
        d, i, name = heapq.heappop(queue)
        if d != distance[i]:
            continue
        for j, cost in adjacency[i]:
            nd = d+cost
            if nd < distance[j]:
                distance[j] = nd; labels[j] = name
                heapq.heappush(queue, (nd, j, name))
    # Isolated degenerate triangles cannot inherit a semantic label safely.
    for i in range(len(labels)):
        if labels[i] is None:
            labels[i] = 'Exclude'
    interfaces = {}
    for a, b, angle, edge in links:
        x, y = labels[a], labels[b]
        if x != y and 'Exclude' not in (x, y):
            key = '|'.join(sorted((x, y)))
            interfaces.setdefault(key, []).append(dict(vertices=list(edge),
                faces=[a, b], bend_degrees=math.degrees(angle)))
    report = dict(method='Reviewed 15-degree crease components; geodesic residual propagation weighted by dihedral angle. No weights, UVs or textures used.',
                  counts={n:labels.count(n) for n in sorted(set(labels))},
                  reviewed_components=REVIEWED, interfaces=interfaces)
    (OUT/'clothing_segmentation.json').write_text(json.dumps(report, indent=2))
    np.savez_compressed(OUT/'clothing_labels.npz', labels=np.array(labels), source_vertex=inv)
    return labels, components, report


def preview(obj):
    import bpy
    labels, components, report = segment(obj)
    colors = {'Vest':(.05,.45,.30), 'Sleeve_L':(.35,.63,.72), 'Sleeve_R':(.35,.63,.72),
              'Cuff_L':(.055,.10,.18), 'Cuff_R':(.055,.10,.18), 'Trousers':(.38,.59,.62),
              'Boot_L':(.03,.045,.10), 'Boot_R':(.03,.045,.10), 'Exclude':(.55,.55,.55)}
    obj.data.materials.clear()
    names = list(colors)
    for name, rgb in colors.items():
        m = bpy.data.materials.new('Audit '+name); m.diffuse_color=(*rgb, 1)
        obj.data.materials.append(m)
    for p, name in zip(obj.data.polygons, labels):
        p.material_index = names.index(name)
    bpy.context.scene.display.shading.color_type = 'MATERIAL'
    print(json.dumps(report['counts']))


def material_regions(obj, labels, components):
    """Trace raised front placket/collar borders with geometric minimum cuts."""
    v,f,c,inv,links,roots=graph(obj.data)
    Cut=runpy.run_path(str(ROOT/'segment_face_geometry.py'))['Cut']
    x,y,z=c.T;vest=np.array(labels)=='Vest'
    regions={}
    definitions={
        'placket':(vest&(abs(x)<.035)&(y<-.02)&(z>.373)&(z<.552),
                   (abs(x)<.006)&(y<-.060)&(z>.385)&(z<.529),
                   (abs(x)>.026)|(z>.542)|(y>-.04)),
        'collar':(vest&(z>.525)&(z<.578),z>.558,z<.541),
    }
    for name,(domain,inside,outside) in definitions.items():
        ids=np.flatnonzero(domain);local={int(fi):i for i,fi in enumerate(ids)}
        cut=Cut(len(ids)+2);s=len(ids);t=s+1
        for fi in ids:
            if inside[fi]:cut.add(s,local[int(fi)],1000)
            if outside[fi]:cut.add(local[int(fi)],t,1000)
        for a,b,theta,edge in links:
            if a not in local or b not in local:continue
            cost=float(np.linalg.norm(c[a]-c[b]))/(1+(theta/.15)**4)+1e-6
            cut.add(local[a],local[b],cost);cut.add(local[b],local[a],cost)
        reachable=cut.solve(s,t)
        regions[name]={int(fi) for fi in ids if local[int(fi)] in reachable}
    return regions


if __name__ == '__main__':
    import bpy
    audit(bpy.data.objects['material'])

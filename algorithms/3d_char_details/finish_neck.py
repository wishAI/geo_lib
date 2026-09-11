"""Finish the welded neck: local surface relaxation, shared normals and color field."""
import json
from pathlib import Path
import bpy
from mathutils import Vector

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/landau_v10'


def run():
    body=bpy.data.objects['Body_Complete'];mesh=body.data
    if body.get('neck_finish'): raise RuntimeError('Neck finish already applied')
    original=[v.co.copy() for v in mesh.vertices]
    normals=[n.vector.copy() for n in mesh.corner_normals]
    neighbors=[set() for v in mesh.vertices]
    for e in mesh.edges:
        a,b=e.vertices;neighbors[a].add(b);neighbors[b].add(a)
    # Taubin relaxation damps residual dents without Laplacian shrinkage.
    # Topology, weights and the shape-key deltas remain unchanged.
    band=[v.index for v in mesh.vertices if .750<v.co.z<.812]
    for _ in range(30):
        for factor in (.5,-.53):
            updates={}
            for i in band:
                t=(original[i].z-.750)/.062;weight=(4*t*(1-t))**2
                mean=sum((mesh.vertices[j].co for j in neighbors[i]),Vector())/len(neighbors[i])
                updates[i]=mesh.vertices[i].co+(mean-mesh.vertices[i].co)*factor*weight
            for i,p in updates.items():mesh.vertices[i].co=p
    for i in band:
        delta=mesh.vertices[i].co-original[i]
        for key in mesh.shape_keys.key_blocks:key.data[i].co+=delta
    mesh.update()
    # Explicit area-weighted normals ignore sharp-edge flags inherited from the
    # old material partitions. All corners of a neck vertex share one normal.
    area_normals=[Vector() for v in mesh.vertices]
    for p in mesh.polygons:
        for i in p.vertices:area_normals[i]+=p.normal*p.area
    result=[]
    for loop,n in zip(mesh.loops,normals):
        p=mesh.vertices[loop.vertex_index].co
        result.append(area_normals[loop.vertex_index].normalized() if .745<p.z<.816 and abs(p.x)<.09 else n)
    mesh.normals_split_custom_set(result)
    color=mesh.color_attributes['FurColor'];colors=[[] for v in mesh.vertices]
    for p in mesh.polygons:
        rgba=mesh.materials[p.material_index].node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value
        for i in p.loop_indices:
            colors[mesh.loops[i].vertex_index].append(Vector(color.data[i].color if p.material_index==0 else rgba))
    colors=[sum(c,Vector((0,0,0,0)))/len(c) if c else Vector((0,0,0,1)) for c in colors]
    paint_band=[v.index for v in mesh.vertices if .73<v.co.z<.818]
    for _ in range(25):
        updates={}
        for i in paint_band:
            z=mesh.vertices[i].co.z
            weight=max(0,min(1,(z-.73)/.02))*max(0,min(1,(.818-z)/.012))
            mean=sum((colors[j] for j in neighbors[i]),Vector((0,0,0,0)))/len(neighbors[i])
            updates[i]=colors[i].lerp(mean,.5*weight)
        for i,c in updates.items():colors[i]=c
    # Continue a single color attribute across both sides of the former join.
    for p in mesh.polygons:
        if any(.73<mesh.vertices[i].co.z<.818 for i in p.vertices):
            p.material_index=0
            for i in p.loop_indices:color.data[i].color=colors[mesh.loops[i].vertex_index]
    mesh.materials[0].node_tree.nodes.get('Principled BSDF').inputs['Roughness'].default_value=.8
    mesh.update()
    protected_error=max((v.co-original[v.index]).length for v in mesh.vertices if original[v.index].z>=.818)
    assert protected_error==0
    stats={'surface':'30 tapered Taubin passes in Z .750–.812; shape deltas retained',
           'normals':'Area-weighted shared neck normals across material partitions',
           'color':'Continuous corner color field on both sides of the old rim; matching roughness',
           'protected_head_max_error':protected_error,'moved_vertices':len(band),
           'max_displacement':max((v.co-original[v.index]).length for v in mesh.vertices)}
    body['neck_finish']=True
    report=json.loads((OUT/'body_build.json').read_text());report['continuous_skin']['neck_finish']=stats
    if (OUT/'body_build.json').is_symlink():(OUT/'body_build.json').unlink()
    (OUT/'body_build.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(stats));return report

if __name__=='__main__':run()

"""Geometry, shared skin, local fitting control and articulated pose evidence."""
from pathlib import Path
import bpy,bmesh,json,math,runpy
import numpy as np
from mathutils import Vector,Quaternion
from mathutils.bvhtree import BVHTree
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs/landau_v10'
GARMENTS=['Vest','Sleeve_L','Sleeve_R','Cuff_L','Cuff_R','Trousers','Boot_L','Boot_R']

def evaluated(o):
    e=o.evaluated_get(bpy.context.evaluated_depsgraph_get());m=e.to_mesh()
    v=np.array([(e.matrix_world@p.co)[:] for p in m.vertices]);f=[p.vertices[:] for p in m.polygons];e.to_mesh_clear();return v,f

def clothing_clearance():
    v,f=evaluated(bpy.data.objects['Body_Complete']);tree=BVHTree.FromPolygons(v.tolist(),f);result={}
    for n in GARMENTS:
        p,faces=evaluated(bpy.data.objects[n]);centers=np.array([p[list(f)].mean(0) for f in faces]);samples=np.concatenate([p,centers]);depths=[]
        for q in samples:
            point,normal,index,dist=tree.find_nearest(Vector(q))
            if (Vector(q)-point).dot(normal)<-.0005:depths.append(dist)
        result[n]={'samples':len(samples),'penetrating_samples':len(depths),'max_penetration':max(depths,default=0)}
    return result

def reset():
    rig=bpy.data.objects['Landau_Rig']
    for p in rig.pose.bones:p.rotation_mode='QUATERNION';p.rotation_quaternion=Quaternion();p.location=Vector();p.scale=Vector((1,1,1))
    for o in bpy.context.scene.objects:
        if o.type=='MESH' and o.data.shape_keys:
            for k in o.data.shape_keys.key_blocks:k.value=0
            o.data.shape_keys.update_tag()
    bpy.context.scene.frame_set(1);bpy.context.view_layer.update()

def rotate(name,axis,degrees):
    rig=bpy.data.objects['Landau_Rig'];p=rig.pose.bones[name];q=(rig.matrix_world@p.bone.matrix_local).to_quaternion()
    p.rotation_mode='QUATERNION';p.rotation_quaternion=q.inverted()@Quaternion(Vector(axis),math.radians(degrees))@q

def run(render=True):
    reset();body=bpy.data.objects['Body_Complete'];bm=bmesh.new();bm.from_mesh(body.data)
    topology={'vertices':len(bm.verts),'faces':len(bm.faces),'boundary_edges':sum(e.is_boundary for e in bm.edges),'nonmanifold_edges':sum(not e.is_manifold for e in bm.edges),'loose_vertices':sum(not v.link_faces for v in bm.verts),'quad_faces':sum(len(f.verts)==4 for f in bm.faces),'neck_boundary_edges':sum(e.is_boundary and all(.752<v.co.z<.81 for v in e.verts) for e in bm.edges)};bm.free()
    if body.get('continuous_head'):assert topology['neck_boundary_edges']==0 and topology['loose_vertices']==0, 'Welded neck must have no open edges'
    else:assert topology['boundary_edges']==82 and topology['nonmanifold_edges']==82 and topology['loose_vertices']==0, 'Expected only the paired neck rim to remain open'
    invalid=[]
    for o in [body]+[bpy.data.objects[n] for n in GARMENTS]:
        assert next(m.object for m in o.modifiers if m.type=='ARMATURE')==bpy.data.objects['Landau_Rig']
        for v in o.data.vertices:
            if len(v.groups)>4 or abs(sum(g.weight for g in v.groups)-1)>1e-4:invalid.append([o.name,v.index])
    assert not invalid
    neutral,faces=evaluated(body);edges=np.array([e.vertices[:] for e in body.data.edges]);length=np.linalg.norm(neutral[edges[:,0]]-neutral[edges[:,1]],axis=1)
    poses={'elbow_90':[('forearm_stretch_l',(1,0,0),90),('forearm_stretch_r',(1,0,0),65)],'raised_arms':[('arm_stretch_l',(0,1,0),-35),('arm_stretch_r',(0,1,0),35)],'bent_legs':[('thigh_stretch_l',(1,0,0),-40),('leg_stretch_l',(1,0,0),80),('thigh_stretch_r',(1,0,0),-15),('leg_stretch_r',(1,0,0),35)],'ankle_25':[('foot_l',(1,0,0),25)],'torso_twist':[('spine_02_x',(0,0,1),20)]}
    result={'topology':topology,'invalid_skin_vertices':len(invalid),'neutral_clothing_clearance':clothing_clearance(),'poses':{}}
    h=runpy.run_path(str(ROOT/'inspect_landau.py'))
    for name,changes in poses.items():
        reset()
        for args in changes:rotate(*args)
        bpy.context.view_layer.update();p,_=evaluated(body);assert np.isfinite(p).all();ratios=np.linalg.norm(p[edges[:,0]]-p[edges[:,1]],axis=1)/np.maximum(length,1e-9)
        result['poses'][name]={'finite':True,'edge_stretch_p99':float(np.percentile(ratios,99)),'edge_stretch_max':float(ratios.max()),'edge_collapse_p01':float(np.percentile(ratios,1))}
        if name in ['elbow_90','bent_legs']:
            result['poses'][name]['clothing_clearance']=clothing_clearance()
            if render:
                for n in GARMENTS:bpy.data.objects[n].hide_render=True
                h['render_view']('body_'+name,(1.3,-3,.8),target=(0,0,.69),scale=1.50)
                for n in GARMENTS:bpy.data.objects[n].hide_render=False
                h['render_view']('dressed_'+name,(1.3,-3,.8),target=(0,0,.69),scale=1.50)
    reset();report=json.loads((OUT/'body_build.json').read_text());controls=report['adjustment_controls'];tested=[]
    for name in controls:
        for o in bpy.context.scene.objects:
            if o.type=='MESH' and o.data.shape_keys and name in o.data.shape_keys.key_blocks:o.data.shape_keys.key_blocks[name].value=1;o.data.shape_keys.update_tag()
        bpy.context.view_layer.update()
        for n in GARMENTS:assert np.isfinite(evaluated(bpy.data.objects[n])[0]).all()
        if controls[name]['kind']=='outfit':assert np.max(np.abs(evaluated(body)[0]-neutral))<1e-6,'Garment control changed body'
        tested.append(name);reset()
    result['all_adjustment_controls_tested_individually_at_one']=tested
    result['scope']='Revision 5 has a welded continuous head/body junction; other pre-existing facial openings are separate components. Original garments are reset for manual fitting; their clearance values are diagnostics, not acceptance passes.'
    (OUT/'body_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));return result
if __name__=='__main__':run()

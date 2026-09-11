"""Integrate the user-provided body into the preserved Landau facial master.
Original garment Basis geometry, rig and head stay at the original scale.
"""
from pathlib import Path
import bpy,bmesh,runpy,json,hashlib,math
import numpy as np
from mathutils import Vector,Matrix
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs/landau_v10'
api=runpy.run_path(str(ROOT/'rebuild_body.py'))
smooth=api['smooth'];GARMENTS=api['GARMENTS']
BONE_MAP={'Hips':'root_x','Spine':'spine_01_x','Spine1':'spine_02_x','Spine2':'spine_03_x','Neck':'neck_x','Head':'head_x'}
for a,b in [('Left','l'),('Right','r')]:
 for x,y in [('Shoulder','shoulder_'),('Arm','arm_stretch_'),('ForeArm','forearm_stretch_'),('Hand','hand_'),('UpLeg','thigh_stretch_'),('Leg','leg_stretch_'),('Foot','foot_'),('ToeBase','toes_01_')]:BONE_MAP[a+x]=y+b

def original_hash(o):
 data={'vertices':[v.co[:] for v in o.data.vertices],'faces':[p.vertices[:] for p in o.data.polygons],'uv':[[v.uv[:] for v in l.data] for l in o.data.uv_layers],'normals':[v.vector[:] for v in o.data.corner_normals],'weights':[[(o.vertex_groups[g.group].name,g.weight) for g in v.groups] for v in o.data.vertices],'materials':[m.name for m in o.data.materials]}
 return hashlib.sha256(json.dumps(data).encode()).hexdigest()

def loops(bm):
 pending={e for e in bm.edges if e.is_boundary};result=[]
 while pending:
  start=pending.pop();vertices=[start.verts[0],start.verts[1]];prev=start
  while vertices[-1]!=vertices[0]:
   candidates=[e for e in vertices[-1].link_edges if e in pending]
   if not candidates:break
   e=candidates[0];pending.remove(e);vertices.append(e.other_vert(vertices[-1]));prev=e
  if vertices[-1]==vertices[0]:result.append(vertices[:-1])
 return result

def neck_target():
 bm=bmesh.new()
 for n in ['Head','Face_Cream']:
  h=bpy.data.objects[n];m=h.data.copy();m.transform(h.matrix_world);bm.from_mesh(m);bpy.data.meshes.remove(m)
 bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=1e-6)
 ring=min(loops(bm),key=lambda vs:min(v.co.z for v in vs))
 result=[v.co.copy() for v in ring];bm.free();return result

SCALE=.79
GROUND=.006648703012615442*SCALE

def align_rig():
 src=bpy.data.objects['FBX_SourceRig'];rig=bpy.data.objects['Landau_Rig']
 def point(name):return (src.matrix_world@src.data.bones[name].head_local)*SCALE+Vector((0,0,GROUND))
 old={b.name:(b.head_local.copy(),b.tail_local.copy()) for b in rig.data.bones}
 head_offset=point('Head')-old['head_x'][0]+Vector((0,0,.025))
 hand_transforms={}
 for side,suffix in [('Left','l'),('Right','r')]:
  h=old['hand_'+suffix];a=point(side+'Hand');axis=point(side+'Hand_end')-a;q=(h[1]-h[0]).normalized().rotation_difference(axis.normalized())
  hand_transforms[suffix]=(h[0],a,q)
 mapping={}
 for n,t in BONE_MAP.items():
  if n=='Head' or 'Hand' in n:continue
  mapping[t]=point(n)
 bpy.ops.object.select_all(action='DESELECT');rig.select_set(True);bpy.context.view_layer.objects.active=rig;bpy.ops.object.mode_set(mode='EDIT')
 for b in rig.data.edit_bones:
  if b.name in ['head_x','ear_l','ear_r']:b.head=old[b.name][0]+head_offset;b.tail=old[b.name][1]+head_offset
  elif b.name.startswith(('hand_','thumb','index','middle','ring','pinky')):
   suffix=b.name[-1];a,c,q=hand_transforms[suffix];b.head=c+q@(old[b.name][0]-a);b.tail=c+q@(old[b.name][1]-a)
 for n,t in [('Hips','root_x'),('Spine','spine_01_x'),('Spine1','spine_02_x'),('Spine2','spine_03_x'),('Neck','neck_x')]:
  nxt={'Hips':'Spine','Spine':'Spine1','Spine1':'Spine2','Spine2':'Neck','Neck':'Head'}[n];rig.data.edit_bones[t].head=point(n);rig.data.edit_bones[t].tail=point(nxt)
 for side,suffix in [('Left','l'),('Right','r')]:
  for target,a,b in [('shoulder_','Shoulder','Arm'),('arm_stretch_','Arm','ForeArm'),('arm_twist_','Arm','ForeArm'),('thigh_stretch_','UpLeg','Leg'),('thigh_twist_','UpLeg','Leg'),('foot_','Foot','ToeBase'),('toes_01_','ToeBase','ToeBase_end')]:
   bone=rig.data.edit_bones[target+suffix];bone.head=point(side+a);bone.tail=point(side+b)
  for prefix,a,b in [('forearm','ForeArm','Hand'),('leg','Leg','Foot')]:
   p=point(side+a);q=point(side+b);mid=(p+q)/2
   rig.data.edit_bones[prefix+'_stretch_'+suffix].head=p;rig.data.edit_bones[prefix+'_stretch_'+suffix].tail=mid
   rig.data.edit_bones[prefix+'_twist_'+suffix].head=mid;rig.data.edit_bones[prefix+'_twist_'+suffix].tail=q
 rig.data.edit_bones['tail_x'].head=point('Hips')+Vector((0,.075,.01));rig.data.edit_bones['tail_x'].tail=rig.data.edit_bones['tail_x'].head+Vector((0,.06,0))
 rig.data.edit_bones['neck_x'].tail.z+=.025
 bpy.ops.object.mode_set(mode='OBJECT')
 for o in bpy.context.scene.objects:
  if o.type!='MESH' or o.name.startswith(('FBX_','Hand_','Body_')) or o.name in GARMENTS or o.name=='Tail':continue
  o.location+=head_offset
 # Garment shapes are unchanged. Only rigid placement follows the new joints;
 # all actual tailoring is exposed as zero-default controls for the user.
 placements={}
 for n in GARMENTS:
  o=bpy.data.objects[n]
  if n.startswith('Boot'):
   suffix=n[-1].lower();delta=rig.data.bones['foot_'+suffix].head_local-old['foot_'+suffix][0];delta.z=0
  elif n.startswith('Cuff'):
   suffix=n[-1].lower();a,c,q=hand_transforms[suffix];delta=c-a
  elif n.startswith('Sleeve'):
   suffix=n[-1].lower();delta=rig.data.bones['arm_stretch_'+suffix].head_local-old['arm_stretch_'+suffix][0]
  else:
   b='spine_03_x' if n=='Vest' else 'root_x';delta=rig.data.bones[b].head_local-old[b][0]
  o.location+=delta;placements[n]=list(delta)
 return head_offset,hand_transforms,placements,old

def trim_and_map():
 source=bpy.data.objects['FBX_SourceBody'];src=bpy.data.objects['FBX_SourceRig'];rig=bpy.data.objects['Landau_Rig']
 o=source.copy();o.data=source.data.copy();o.name='Body_Complete';bpy.context.scene.collection.objects.link(o);o.parent=None;o.matrix_world=Matrix.Identity(4)
 for m in list(o.modifiers):o.modifiers.remove(m)
 bm=bmesh.new();bm.from_mesh(o.data)
 for v in bm.verts:v.co=(source.matrix_world@v.co)*SCALE+Vector((0,0,GROUND))
 bmesh.ops.bisect_plane(bm,geom=list(bm.verts)+list(bm.edges)+list(bm.faces),plane_co=(0,0,.96*SCALE+GROUND),plane_no=(0,0,1),clear_outer=True,dist=1e-7)
 for side,sign in [('Left',1),('Right',-1)]:
  wrist=(src.matrix_world@src.data.bones[side+'Hand'].head_local)*SCALE+Vector((0,0,GROUND));elbow=(src.matrix_world@src.data.bones[side+'ForeArm'].head_local)*SCALE+Vector((0,0,GROUND));axis=(wrist-elbow).normalized()
  vs={v for v in bm.verts if sign*v.co.x>.15 and v.co.z>.30};geom=list(vs)+[e for e in bm.edges if all(v in vs for v in e.verts)]+[f for f in bm.faces if all(v in vs for v in f.verts)]
  bmesh.ops.bisect_plane(bm,geom=geom,plane_co=wrist,plane_no=axis,clear_outer=True,dist=1e-7)
 pending=set(bm.verts);components=[]
 while pending:
  st=[pending.pop()];component=[]
  while st:
   v=st.pop();component.append(v)
   for e in v.link_edges:
    q=e.other_vert(v)
    if q in pending:pending.remove(q);st.append(q)
  components.append(component)
 main=max(components,key=len);bmesh.ops.delete(bm,geom=[v for c in components if c is not main for v in c],context='VERTS');bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));bm.to_mesh(o.data);bm.free()
 bpy.ops.object.select_all(action='DESELECT');o.select_set(True);bpy.context.view_layer.objects.active=o
 mod=o.modifiers.new('Supplied sculpt practical density','DECIMATE');mod.ratio=.28;bpy.ops.object.modifier_apply(modifier=mod.name)
 for g in o.vertex_groups:
  if g.name in BONE_MAP:g.name=BONE_MAP[g.name]
 weights=[{o.vertex_groups[g.group].name:g.weight for g in v.groups} for v in o.data.vertices];adj=[[] for v in weights]
 for e in o.data.edges:
  a,b=e.vertices;adj[a].append(b);adj[b].append(a)
 for _ in range(10):
  result=[]
  for i,w in enumerate(weights):
   d={n:v*.6 for n,v in w.items()}
   for j in adj[i]:
    for n,v in weights[j].items():d[n]=d.get(n,0)+v*.4/len(adj[i])
   result.append(d)
  weights=result
 api['assign'](o,weights)
 for p in o.data.polygons:p.use_smooth=True
 for e in o.data.edges:e.use_edge_sharp=False
 o.data.normals_split_custom_set([(0,0,0)]*len(o.data.loops));source.hide_render=True
 if source.name in bpy.context.view_layer.objects:source.hide_set(True)
 return o,{'method':'One uniform scale on the supplied body; original rig joints repositioned to source landmarks','uniform_scale_xyz':[SCALE]*3,'ground_translation':GROUND,'source_shape_preserved_except_seams_and_decimation':True,'joints':[{'source':n,'target':t,'target_position':list(rig.data.bones[t].head_local)} for n,t in BONE_MAP.items()]}

def join_boundaries(o,hand_transforms):
 for suffix,(a,c,q) in hand_transforms.items():
  hand=bpy.data.objects['Hand_'+suffix.upper()]
  for v in hand.data.vertices:v.co=c+q@(v.co-a)
 # Exact original neck boundary comes from the head + cream face partition.
 target=neck_target();bm=bmesh.new();bm.from_mesh(o.data);bm.verts.ensure_lookup_table()
 rings=loops(bm);print('Body open rings',[(len(r),tuple(sum((v.co for v in r),Vector())/len(r))) for r in rings])
 data=[[(v.index,v.co.copy()) for v in r] for r in rings];bm.free()
 cage=api['Cage']();cage.v=[v.co[:] for v in o.data.vertices];cage.f=[list(p.vertices) for p in o.data.polygons];cage.mat=[0]*len(cage.f)
 cage.ws=[{o.vertex_groups[g.group].name:g.weight for g in v.groups} for v in o.data.vertices]
 joints=[]
 for ring in data:
  center=sum((p for _,p in ring),Vector())/len(ring);a=[i for i,_ in ring]
  if abs(center.x)<.09:
   lookup={tuple(round(t,6) for t in (h.matrix_world@v.co)):{h.vertex_groups[g.group].name:g.weight for g in v.groups} for h in [bpy.data.objects['Head'],bpy.data.objects['Face_Cream']] for v in h.data.vertices}
   b=[cage.point(p,w=lookup[tuple(round(t,6) for t in p)]) for p in target];b=cage.align(a,b)
   # A short collar blends toward the preserved, irregular original neck rim.
   cage.bridge(a,b);joints.append({'name':'neck','paired_vertices':len(b),'gap':0.0})
  else:
   side='L' if center.x>0 else 'R';hand=bpy.data.objects['Hand_'+side];offset=len(cage.v)
   for v in hand.data.vertices:cage.point(v.co,w={hand.vertex_groups[g.group].name:g.weight for g in v.groups})
   for f in hand.data.polygons:cage.face([offset+i for i in f.vertices])
   b=[offset+i for i in api['boundary_loop'](hand.data)];b=cage.align(a,b);cage.bridge(a,b)
   joints.append({'name':'wrist_'+side,'paired_vertices':len(b),'gap':0.0});bpy.data.objects.remove(hand,do_unlink=True)
 mesh=bpy.data.meshes.new('User FBX body + original hands + transition collars');mesh.from_pydata(cage.v,[],cage.f);mesh.update();o.data=mesh
 api['bind'](o,bpy.data.objects['Landau_Rig'],cage.ws)
 bm=bmesh.new();bm.from_mesh(mesh);bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));bm.to_mesh(mesh);bm.free()
 for p in mesh.polygons:p.use_smooth=True
 return joints

def paint_body(o):
 o.data.materials.clear();m=bpy.data.materials.new('Supplied body · blue fur / cream front');m.use_nodes=True
 p=m.node_tree.nodes.get('Principled BSDF');p.inputs['Roughness'].default_value=.78
 attr=o.data.color_attributes.new(name='FurColor',type='FLOAT_COLOR',domain='CORNER');node=m.node_tree.nodes.new('ShaderNodeVertexColor');node.layer_name='FurColor';m.node_tree.links.new(node.outputs['Color'],p.inputs['Base Color'])
 blue=bpy.data.materials['Fur blue'].node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value[:]
 cream=bpy.data.materials['Cream face'].node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value[:]
 widths=[(.55,.004),(.61,.047),(.68,.058),(.75,.045),(.82,.070),(.88,.080),(.92,.036),(.97,.037)]
 for loop in o.data.loops:
  v=o.data.vertices[loop.vertex_index];x,y,z=v.co;z=(z-GROUND)/SCALE;ax=abs(x)/SCALE;y/=SCALE
  width=float(np.interp(z,[a for a,b in widths],[b for a,b in widths]));w=(1-smooth(width-.0015,width+.0015,ax))*(1-smooth(.004,.020,y))*smooth(.55,.60,z)
  if ax>.18 and z<.70:w=max(w,smooth(.18,.22,ax)*(1-smooth(.61,.69,z)))
  if z<.13:w=max(w,1-smooth(.095,.12,z))
  attr.data[loop.index].color=tuple(a*(1-w)+b*w for a,b in zip(blue,cream))
 o.data.materials.append(m)

def run():
 before=api['face_snapshot']();before={n:v for n,v in before.items() if not n.startswith('FBX_')}
 clothing={n:original_hash(bpy.data.objects[n]) for n in GARMENTS}
 api['repair_hands']()
 head_offset,hand_transforms,placements,old=align_rig()
 body,landmarks=trim_and_map();seams=join_boundaries(body,hand_transforms);paint_body(body)
 for n in ['Body_UnderClothes','Tail','FBX_SourceBody','FBX_SourceRig']:
  if n in bpy.data.objects:bpy.data.objects.remove(bpy.data.objects[n],do_unlink=True)
 body['part_type']='inferred_body';body['default_hidden']=False;body['complete_under_outfit']=True;body['body_revision']=4;body['head_rigid_lift']=head_offset.z;body['source']='User-provided landau_body.fbx'
 body['anatomy_method']='Joint-matched supplied FBX sculpt with original Landau hands and preserved head'
 for o in bpy.context.scene.objects:
  if o.type=='MESH':o.hide_render=bool(o.get('default_hidden'));o.hide_set(bool(o.get('default_hidden')))
 after=api['face_snapshot']();assert before==after,'Protected facial local data changed'
 # Reversible rigid placement is recorded separately from original garment shape.
 for n in GARMENTS:
  o=bpy.data.objects[n];o.location-=Vector(placements[n]);assert original_hash(o)==clothing[n],n+' source garment changed';o.location+=Vector(placements[n])
 result={'revision':4,'reference':'inputs/landau_v10/internal_reference.png','source_fbx':'inputs/landau_v10/landau_body.fbx','source_fbx_sha256':hashlib.sha256((ROOT/'inputs/landau_v10/landau_body.fbx').read_bytes()).hexdigest(),'head_rigid_lift':head_offset.z,'head_rigid_translation':list(head_offset),'garment_rigid_placement':placements,'body_mesh':body.name,'garments':list(GARMENTS),'method':body['anatomy_method'],'face_exactly_preserved':True,'face_hashes_before':before,'face_hashes_after':after,'original_clothing_hashes':clothing,'original_clothing_exactly_reset':True,'joint_alignment':landmarks,'transition_seams':seams}
 (OUT/'body_build.json').write_text(json.dumps(result,indent=2));print(json.dumps({'joints':len(landmarks['joints']),'seams':seams,'body_vertices':len(body.data.vertices)}));return result
if __name__=='__main__':run()

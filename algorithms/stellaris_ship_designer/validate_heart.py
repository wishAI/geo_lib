"""Compare native PDX round-trip part bounds and horn positions across clips."""
import bpy,sys,json,argparse
from pathlib import Path
from mathutils import Vector
parser=argparse.ArgumentParser()
parser.add_argument('--blend',required=True)
parser.add_argument('--native',required=True)
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
native=Path(args.native).resolve()
design=json.loads((Path(__file__).parent/'inputs/stellaris_heart.json').read_text())
hidden_parts={'teeth','tongue'} if design['ship'].get('hideMouthInterior',True) else set()
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'helper_repos'))
from io_pdx_mesh.pdx_blender.blender_import_export import import_meshfile,import_animfile
bpy.ops.wm.open_mainfile(filepath=str(Path(args.blend).resolve()));rig=bpy.data.objects['Stellaris_Heart_Rig']
def bounds():
 result={};dg=bpy.context.evaluated_depsgraph_get()
 for obj in bpy.context.scene.objects:
  if obj.type!='MESH':continue
  if obj.name in hidden_parts or obj.data.name in hidden_parts:continue
  ev=obj.evaluated_get(dg);mesh=ev.to_mesh();pts=[obj.matrix_world@v.co for v in mesh.vertices];result[obj.data.name]=[min(v[i] for v in pts) for i in range(3)]+[max(v[i] for v in pts) for i in range(3)];ev.to_mesh_clear()
 result['horn_muzzle']=list(bpy.data.objects['horn_muzzle'].matrix_world.translation)
 return result
reference={}
clips = [(name, int(bpy.data.actions[name].frame_range[1])) for name in ['idle', 'moving', 'dash', 'planet_killer', 'moving_va', 'moving_genmaxx', 'moving_argodaemon']]
for name,end in clips:
 rig.animation_data.action=bpy.data.actions[name]
 for f in [1,(end+1)//2,end]:bpy.context.scene.frame_set(f);reference[name,f]=bounds()
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
# Remove orphan names so round-trip mesh names can be matched exactly.
for mesh in list(bpy.data.meshes):
 if mesh.users==0:bpy.data.meshes.remove(mesh)
import_meshfile(str(native/'stellaris_heart.mesh'),use_diffuse_alpha=True,diffuse_alpha_mode='BLEND')
rig=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE');bpy.context.view_layer.objects.active=rig
errors=[]
for name,end in clips:
 import_animfile(str(native/('heart_'+name+'.anim')))
 for f in [1,(end+1)//2,end]:
  bpy.context.scene.frame_set(f);actual=bounds()
  for key,expected in reference[name,f].items():
   err=max(abs(a-b) for a,b in zip(actual[key],expected));errors.append((err,name,f,key))
errors.sort(reverse=True);print('ROUNDTRIP',errors[:8]);(native/'roundtrip.json').write_text(json.dumps({'maxBoundsOrLocatorError':errors[0][0],'samples':len(errors),'worst':errors[:8],'passed':errors[0][0]<.002,'gameRuntimeTested':False},indent=2))
assert errors[0][0]<.002

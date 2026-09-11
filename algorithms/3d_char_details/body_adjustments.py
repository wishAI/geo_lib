"""Independent body shaping and original-garment tailoring, all neutral at zero."""
import bpy,math,json,runpy
from pathlib import Path
from mathutils import Vector
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs/landau_v10'
api=runpy.run_path(str(ROOT/'rebuild_body.py'));smooth=api['smooth']

def add(o,name,label,group,kind,fn,minimum=-1):
 if not o.data.shape_keys:o.shape_key_add(name='Basis')
 if name in o.data.shape_keys.key_blocks:o.shape_key_remove(o.data.shape_keys.key_blocks[name])
 key=o.shape_key_add(name=name);key.slider_min=minimum;key.slider_max=1;key.value=0
 for v in o.data.vertices:key.data[v.index].co=v.co+fn(v)
 o.data.shape_keys.update_tag()
 return {'label':label,'group':group,'kind':kind,'min':minimum,'max':1}

def run():
 meta={};body=bpy.data.objects['Body_Complete'];rig=bpy.data.objects['Landau_Rig'];S=.79;G=.0052524753799662
 def gaussian(z,c,w):return math.exp(-2*((z-c)/w)**2)
 def body_delta(v,name):
  p=v.co;z=(p.z-G)/S;sign=1 if p.x>0 else -1;ax=abs(p.x);d=Vector();front=1-smooth(.005,.035,p.y)
  # Locks at the original neck rim and both original hand joins.
  lock=1-smooth(.720,.746,p.z)
  if name.startswith('bodyNeck'):
   w=gaussian(p.z,.728,.027)*lock;d.x=.012*sign*w;s=0
  elif name in ['bodyUpperArmVolume','bodyForearmVolume']:
   suffix='l' if sign>0 else 'r';a=rig.data.bones['arm_stretch_'+suffix].head_local;b=rig.data.bones['hand_'+suffix].head_local;axis=b-a;t=(p-a).dot(axis)/axis.length_squared;center=a+axis*max(0,min(1,t));radial=p-center
   if ax>.065 and radial.length<.07 and 0<t<1:
    w=gaussian(t,.25 if name=='bodyUpperArmVolume' else .68,.30)*(1-smooth(.83,.96,t));d=radial.normalized()*.013*w
  elif name in ['bodyFootWidth','bodyFootLength','bodyHeelVolume']:
   suffix='l' if sign>0 else 'r';a=rig.data.bones['foot_'+suffix].head_local;w=1-smooth(.085,.12,p.z)
   if name=='bodyFootWidth':d.x=(p.x-a.x)*.25*w
   elif name=='bodyFootLength':d.y=(p.y-a.y)*.25*w
   else:d.y=.012*smooth(a.y-.01,a.y+.025,p.y)*w
  elif name=='bodyTailSize':
   if p.y>.068 and .43<p.z<.60:d=(p-Vector((0,.083,.505)))*.25*smooth(.068,.10,p.y)
  else:
   centers={'bodyShoulderWidth':(.89,.085),'bodyChestWidth':(.845,.12),'bodyChestDepth':(.845,.12),'bodyBustVolume':(.835,.065),'bodyBustHeight':(.835,.065),'bodyWaistWidth':(.755,.08),'bodyWaistDepth':(.755,.08),'bodyBellyDepth':(.70,.10),'bodyHipWidth':(.625,.14),'bodyHipDepth':(.625,.14),'bodyThighVolume':(.48,.20),'bodyCalfVolume':(.23,.20)}
   c,wid=centers[name];w=gaussian(z,c,wid)*lock
   if name in ['bodyThighVolume','bodyCalfVolume']:
    suffix='l' if sign>0 else 'r';bone=rig.data.bones[('thigh_stretch_' if name=='bodyThighVolume' else 'leg_stretch_')+suffix];a=bone.head_local;b=bone.tail_local;t=max(0,min(1,(p-a).dot(b-a)/(b-a).length_squared));radial=p-(a+(b-a)*t);radial.z=0
    if radial.length:d=radial.normalized()*.014*w
   elif name=='bodyBustHeight':d.z=.014*w*front
   elif name=='bodyBustVolume':d.y=-.020*w*front;d.x=sign*.006*w*front
   elif name.endswith('Width'):d.x=sign*.016*w*smooth(.005,.04,ax)
   elif name.endswith('Depth'):d.y=(1 if p.y>.01 else -1)*.016*w
   if ax>.10 and z>.72:d*=1-smooth(.10,.13,ax)
  if p.z>.30 and ax>.14:
   suffix='l' if sign>0 else 'r';bone=rig.data.bones['hand_'+suffix];t=(p-bone.head_local).dot((bone.tail_local-bone.head_local).normalized())
   d*=1-smooth(-.025,0,t)
  return d
 body_names={'bodyShoulderWidth':'Shoulder width','bodyChestWidth':'Chest width','bodyChestDepth':'Chest depth','bodyBustVolume':'Bust volume','bodyBustHeight':'Bust height','bodyWaistWidth':'Waist width','bodyWaistDepth':'Waist depth','bodyBellyDepth':'Abdomen depth','bodyHipWidth':'Hip width','bodyHipDepth':'Hip depth','bodyThighVolume':'Thigh volume','bodyCalfVolume':'Calf volume','bodyUpperArmVolume':'Upper arm volume','bodyForearmVolume':'Forearm volume','bodyNeckWidth':'Neck width','bodyFootWidth':'Paw width','bodyFootLength':'Paw length','bodyHeelVolume':'Heel volume','bodyTailSize':'Tail size'}
 for n,label in body_names.items():meta[n]=add(body,n,label,'Body proportions','body',lambda v,n=n:body_delta(v,n))
 labels={'vestChestWidth':'Chest width','vestChestDepth':'Chest depth','vestWaistWidth':'Waist width','vestWaistDepth':'Waist depth','vestShoulderWidth':'Shoulder width','vestLength':'Vest length','skirtFlare':'Skirt flare','vestRaise':'Move vest up / down','vestForward':'Move vest forward / back','sleeveUpperRoom':'Upper sleeve room','sleeveForearmRoom':'Forearm sleeve room','sleeveLength':'Sleeve length','sleeveSpread':'Move sleeves outward','sleeveRaise':'Move sleeves up / down','sleeveForward':'Move sleeves forward / back','cuffOpening':'Cuff opening','cuffWidth':'Cuff width','cuffDepth':'Cuff depth','cuffSlide':'Slide cuffs along wrist','trouserWaistWidth':'Waist width','trouserWaistDepth':'Waist depth','trouserRise':'Waist-to-crotch length','trouserThighRoom':'Thigh width','trouserThighDepth':'Thigh depth','trouserCalfRoom':'Calf width','trouserCalfDepth':'Calf depth','trouserLength':'Leg length','trouserRaise':'Move trousers up / down','trouserForward':'Move trousers forward / back','bootWidth':'Toe-box width','bootLength':'Toe-box length','bootInstep':'Instep height','bootHeelDepth':'Heel room','bootShaftWidth':'Shaft width','bootShaftDepth':'Shaft depth','bootShaftHeight':'Shaft height','bootRaise':'Move boots up / down','bootForward':'Move boots forward / back','bootSpread':'Move boots outward'}
 for name in api['GARMENTS']:
  o=bpy.data.objects[name]
  if name=='Vest':choices=[n for n in labels if n.startswith('vest') or n=='skirtFlare'];group='Vest'
  elif name.startswith('Sleeve'):choices=[n for n in labels if n.startswith('sleeve')]+['vestShoulderWidth'];group='Sleeves'
  elif name.startswith('Cuff'):choices=[n for n in labels if n.startswith('cuff')]+['sleeveLength'];group='Cuffs'
  elif name=='Trousers':choices=[n for n in labels if n.startswith('trouser')];group='Trousers'
  else:choices=[n for n in labels if n.startswith('boot')];group='Original boots'
  def delta(v,control):
   p=v.co;x,y,z=p;sign=1 if x>0 else -1;d=Vector()
   if control.endswith('Raise'):d.z=.12 if control=='trouserRaise' else .075
   elif control.endswith('Forward'):d.y=-.10 if control=='bootForward' else -.07
   elif control.endswith('Spread'):d.x=sign*.065
   elif control.startswith('vestChest'):
    w=gaussian(z,.49,.075);d.x=.03*sign*w if control.endswith('Width') else 0;d.y=-.04*w*(1-smooth(-.005,.02,y)) if control.endswith('Depth') else 0
   elif control.startswith('vestWaist'):
    w=gaussian(z,.37,.065);d.x=.025*sign*w if control.endswith('Width') else 0;d.y=(1 if y>.005 else -1)*.025*w if control.endswith('Depth') else 0
   elif control=='vestShoulderWidth':d.x=.04*sign*smooth(.44,.53,z)
   elif control=='vestLength':d.z=-.16*(1-smooth(.37,.50,z))
   elif control=='skirtFlare':d=Vector((sign*.05,(1 if y>.005 else -1)*.03,0))*(1-smooth(.28,.37,z))
   elif control.startswith('sleeve') or control.startswith('cuff'):
    a=Vector((sign*.06395,.00994,.52382));b=Vector((sign*.17565,-.002624,.381775));axis=b-a;t=max(0,min(1,(p-a).dot(axis)/axis.length_squared));radial=p-(a+axis*t)
    if control=='sleeveLength':d=axis.normalized()*.075*smooth(.1,.9,t)
    elif control=='cuffSlide':d=axis.normalized()*.06
    elif radial.length:
     w=gaussian(t,.3 if control=='sleeveUpperRoom' else .7,.35) if control.startswith('sleeve') else 1
     d=radial.normalized()*.025*w
     if control=='cuffWidth':d.y=0;d.z=0
     if control=='cuffDepth':d.x=0;d.z=0
   elif control.startswith('trouser'):
    if control=='trouserRise':d.z=.10*smooth(.27,.35,z)
    elif control=='trouserLength':d.z=-.25*(1-smooth(.12,.31,z))
    else:
     w=gaussian(z,.34 if 'Waist' in control else .27 if 'Thigh' in control else .17,.08)
     if control.endswith('Depth'):d.y=(1 if y>-.01 else -1)*.04*w
     else:d.x=sign*.04*w if 'Waist' in control else (1 if x-sign*.06>0 else -1)*.04*w
   elif control.startswith('boot'):
    if control=='bootWidth':d.x=(1 if x-sign*.060>0 else -1)*.035*(1-smooth(.06,.10,z))*smooth(.001,.022,abs(x-sign*.060))
    elif control=='bootLength':d.y=-.14*(1-smooth(-.09,-.010,y))*(1-smooth(.06,.10,z))
    elif control=='bootInstep':d.z=.060*smooth(.004,.025,z)*(1-smooth(.065,.11,z))*(1-smooth(-.02,.015,y))
    elif control=='bootHeelDepth':d.y=.035*smooth(0,.03,y)*(1-smooth(.065,.11,z))
    elif control=='bootShaftWidth':d.x=(x-sign*.055)*1.0*smooth(.04,.09,z)
    elif control=='bootShaftDepth':d.y=(y+.01)*1.0*smooth(.04,.09,z)
    elif control=='bootShaftHeight':d.z=.16*smooth(.04,.12,z)
   return d
  for n in choices:
   minimum=-1 if n.endswith(('Raise','Forward','Spread')) or n=='cuffSlide' else 0
   meta[n]=add(o,n,labels[n],group,'outfit',lambda v,n=n:delta(v,n),minimum)
  o['outfit_controls']=choices;o['default_hidden']=True;o.hide_set(True);o.hide_render=True
 report=json.loads((OUT/'body_build.json').read_text());report['adjustment_controls']=meta;report['outfit_controls']={n:v['label'] for n,v in meta.items() if v['kind']=='outfit'};report['body_controls']={n:v['label'] for n,v in meta.items() if v['kind']=='body'};report['outfit_reset']='Original garment geometry and materials with rigid joint placement; all tailoring controls start at zero. Garments hidden by default for manual fitting.'
 (OUT/'body_build.json').write_text(json.dumps(report,indent=2));print('Created controls',len(report['body_controls']),len(report['outfit_controls']));return report
if __name__=='__main__':run()

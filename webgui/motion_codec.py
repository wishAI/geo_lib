"""Lossless AVP recording representation; paired with browser motion-codec.js."""
import copy
import math

FORMAT='avp-motion-clip-packed'


def _number(v):
    return type(v) in (int,float) and math.isfinite(v)


def _matrix(m):
    return [v for row in (m[:3] if m[3]==[0,0,0,1] else m) for v in row]


def _unmatrix(m):
    if not isinstance(m,list) or len(m) not in (12,16) or not all(_number(v) for v in m):
        raise ValueError('Invalid packed matrix')
    return [m[:4],m[4:8],m[8:12],m[12:] if len(m)==16 else [0,0,0,1]]


def _pack_tracking(tracking):
    if tracking is None:return None
    result={}
    for key,value in tracking.items():
        arm=key.replace('_wrist','_arm')
        result[key]=('arm0' if key.endswith('_wrist') and tracking.get(arm) and value==tracking[arm][0]
                     else [_matrix(m) for m in value] if key.endswith('_arm') else _matrix(value))
    return result


def _unpack_tracking(tracking):
    if tracking is None:return None
    if not isinstance(tracking,dict):raise ValueError('Invalid packed tracking')
    result={}
    for key,value in tracking.items():
        if value=='arm0':continue
        if key in ('left_arm','right_arm'):
            if not isinstance(value,list) or len(value) not in (25,27):raise ValueError('Invalid packed hand')
            result[key]=[_unmatrix(m) for m in value]
        elif key in ('head','left_wrist','right_wrist'):result[key]=_unmatrix(value)
        else:raise ValueError('Unknown tracking field')
    for key,value in tracking.items():
        if value=='arm0':
            arm=key.replace('_wrist','_arm')
            if not key.endswith('_wrist') or arm not in result:raise ValueError('Invalid wrist alias')
            result[key]=copy.deepcopy(result[arm][0])
    return result


def _derived(raw,alignment):
    if raw is None or alignment is None:return None
    def multiply(b):
        a=alignment
        return [[a[r][0]*b[0][c]+a[r][1]*b[1][c]+a[r][2]*b[2][c]+a[r][3]*b[3][c]
                 for c in range(4)] for r in range(4)]
    return {k:[multiply(m) for m in v] if k.endswith('_arm') else multiply(v) for k,v in raw.items()}


def pack(clip):
    if clip.get('format')==FORMAT:return clip
    metadata={k:v for k,v in clip.items() if k!='frames'}
    frames=clip['frames'];keys=sorted({k for f in frames for k in f['pose']})
    constant={k:frames[0]['pose'][k] for k in keys if all(k in f['pose'] and f['pose'][k]==frames[0]['pose'].get(k) for f in frames)}
    dynamic=[k for k in keys if k not in constant];alignments=[];result=[]
    for f in frames:
        item={**f,'pose':[f['pose'].get(k) for k in dynamic]}
        if 'alignment' in f:
            alignment=None if f['alignment'] is None else _matrix(f['alignment'])
            if alignment not in alignments:alignments.append(alignment)
            item['alignment']=alignments.index(alignment)
        if 'rawTracking' in f:item['rawTracking']=_pack_tracking(f['rawTracking'])
        if 'tracking' in f:
            rebuilt=_derived(f.get('rawTracking'),f.get('alignment'))
            item['tracking']='derived' if rebuilt is not None and rebuilt==f['tracking'] else _pack_tracking(f['tracking'])
        result.append(item)
    return dict(format=FORMAT,version=1,metadata=metadata,poseKeys=dynamic,constantPose=constant,alignments=alignments,frames=result)


def unpack(value):
    if not isinstance(value,dict) or value.get('format')!=FORMAT:return value
    metadata=value.get('metadata');frames=value.get('frames');keys=value.get('poseKeys');alignments=value.get('alignments');constant=value.get('constantPose')
    if (value.get('version')!=1 or not isinstance(metadata,dict) or metadata.get('format')!='avp-motion-clip' or
        metadata.get('version')!=1 or not isinstance(frames,list) or not 1<=len(frames)<=1800 or
        not isinstance(keys,list) or len(keys)>256 or not all(isinstance(k,str) for k in keys) or len(set(keys))!=len(keys) or
        not isinstance(alignments,list) or len(alignments)>1800 or not isinstance(constant,dict) or
        len(constant)+len(keys)>256 or not all(_number(v) for v in constant.values()) or set(keys)&set(constant)):
        raise ValueError('Invalid packed recording structure')
    alignments=[None if m is None else _unmatrix(m) for m in alignments];result=[]
    for f in frames:
        if not isinstance(f,dict) or not isinstance(f.get('pose'),list) or len(f['pose'])!=len(keys):raise ValueError('Invalid packed pose')
        pose=dict(constant)
        for k,v in zip(keys,f['pose']):
            if v is not None:
                if not _number(v):raise ValueError('Invalid packed pose value')
                pose[k]=v
        item={**f,'pose':pose}
        if 'alignment' in f:
            index=f['alignment']
            if type(index) is not int or not 0<=index<len(alignments):raise ValueError('Invalid alignment index')
            item['alignment']=copy.deepcopy(alignments[index])
        if 'rawTracking' in f:item['rawTracking']=_unpack_tracking(f['rawTracking'])
        if 'tracking' in f:
            item['tracking']=_derived(item.get('rawTracking'),item.get('alignment')) if f['tracking']=='derived' else _unpack_tracking(f['tracking'])
            if f['tracking']=='derived' and item['tracking'] is None:raise ValueError('Incomplete derived tracking')
        result.append(item)
    return {**metadata,'frames':result}
